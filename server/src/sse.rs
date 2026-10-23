//! Server-sent event streams: the producer half handed to the application
//! ([`StreamSink`]) and the consumer half that becomes the response body
//! ([`SseBody`]).
//!
//! A push frames the event and queues it in a bounded channel: a memcpy, no
//! syscall. The body drains everything already queued (up to
//! [`COALESCE_CAP`]) into one frame on each wakeup, so under load N events
//! cost one socket write instead of N.
//!
//! Waking the body is the one costly step of a push: when the IO threads are
//! parked it is a futex syscall (and, on a VM, an inter-processor interrupt),
//! paid by a producer that holds the GIL. So producers hand the body's waker
//! to a [`Flusher`] instead of waking it. An idle flusher is woken at once and
//! wakes the body at once, so a push to a quiet server is written with no
//! delay; while pushes keep coming the flusher wakes the bodies in batches
//! every tick (1 ms by default) from an IO thread, and producers pay no
//! syscall at all. A tick of zero wakes each body from the producer, as a
//! plain channel does.

use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use hyper::body::{Body, Frame, SizeHint};
use inferencekey_core::domain::sse::{format_data_event, DONE_EVENT, KEEPALIVE_EVENT};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::{mpsc, oneshot, Notify};

use crate::stats::{self, Stats};

/// Queued events per stream before a producer has to wait.
pub const CHANNEL_CAPACITY: usize = 256;

/// The most bytes one coalesced frame gathers (it may exceed this by the last
/// event taken).
pub const COALESCE_CAP: usize = 64 * 1024;

/// Default [`Flusher`] tick: the most a push waits for its body to be woken
/// while the server is busy.
pub const DEFAULT_FLUSH_TICK: Duration = Duration::from_millis(1);

/// Wakes stream bodies on behalf of producers, in batches while busy (see the
/// module documentation).
pub(crate) struct Flusher {
    tick: Duration,
    /// Body wakers handed over since the last batch.
    pending: Mutex<Vec<Waker>>,
    /// The flusher task is running batches (not waiting on `notify`).
    active: AtomicBool,
    notify: Notify,
    stats: Arc<Stats>,
}

impl Flusher {
    /// A flusher whose [`Flusher::run`] task must be spawned on the runtime
    /// (unless `tick` is zero: then every wake happens on the producer).
    pub fn new(tick: Duration, stats: Arc<Stats>) -> Arc<Self> {
        Arc::new(Flusher {
            tick,
            pending: Mutex::new(Vec::new()),
            active: AtomicBool::new(false),
            notify: Notify::new(),
            stats,
        })
    }

    /// Wake each body from its producer (no task needed).
    #[cfg(test)]
    pub fn direct() -> Arc<Self> {
        Self::new(Duration::ZERO, Arc::new(Stats::default()))
    }

    fn wake(&self, waker: Waker) {
        if self.tick.is_zero() {
            waker.wake();
            return;
        }
        lock(&self.pending).push(waker);
        // Only the push that finds the flusher idle pays for waking it.
        if !self.active.swap(true, Ordering::SeqCst) {
            self.notify.notify_one();
        }
    }

    /// The flusher task: wake the first batch at once, then one batch per
    /// tick until a tick finds nothing to wake.
    pub async fn run(self: Arc<Self>) {
        if self.tick.is_zero() {
            return;
        }
        loop {
            self.notify.notified().await;
            loop {
                let batch = std::mem::take(&mut *lock(&self.pending));
                if batch.is_empty() {
                    self.active.store(false, Ordering::SeqCst);
                    // A producer that queued a waker after the take, while
                    // `active` was still set, did not notify: take it here
                    // (unless another producer already flipped `active` and
                    // notified, which runs the loop again).
                    if lock(&self.pending).is_empty() || self.active.swap(true, Ordering::SeqCst) {
                        break;
                    }
                    continue;
                }
                stats::inc(&self.stats.flush_batches);
                stats::add(&self.stats.flush_wakes, batch.len() as u64);
                for waker in batch {
                    waker.wake();
                }
                tokio::time::sleep(self.tick).await;
            }
        }
    }
}

/// How the application answers a streaming route, decided before any byte of
/// the response is written.
#[derive(Debug)]
pub(crate) enum Head {
    /// A plain JSON response (no stream).
    Reply { status: u16, body: Bytes },
    /// A 200 event stream; the body follows from the channel.
    Start,
}

/// The result of a non-blocking send attempt.
#[derive(Debug)]
pub enum SendOutcome {
    /// Queued.
    Sent,
    /// The channel is full (a slow client). Finish the send with
    /// [`StreamSink::wait_send`], which blocks; a Python binding calls it with
    /// the GIL released.
    Full(PendingFrame),
    /// The client is gone, the stream fell too far behind, or the server is
    /// shutting down. Stop producing.
    Closed,
}

/// A framed event that did not fit in the channel yet.
#[derive(Debug)]
pub struct PendingFrame {
    frame: Bytes,
    /// The terminal `[DONE]` event, which is still delivered while the
    /// server drains.
    done: bool,
}

/// Shared state between a sink and its body.
struct SinkShared {
    head: Mutex<Option<oneshot::Sender<Head>>>,
    tx: Mutex<Option<mpsc::Sender<Bytes>>>,
    /// The body's waker while it waits for an event. The body never registers
    /// with the channel itself, so a send does not wake it: the producer takes
    /// this waker and hands it to the flusher.
    waker: Mutex<Option<Waker>>,
    flusher: Arc<Flusher>,
    /// Set by the body's drop (client gone, write failed), a failed send, the
    /// byte bound, or [`StreamSink::finish`].
    closed: AtomicBool,
    /// Set when the byte bound was exceeded: the body then aborts the
    /// connection instead of draining what is queued.
    aborted: AtomicBool,
    /// Bytes queued in the channel and not yet taken by the body.
    queued_bytes: AtomicUsize,
    max_bytes: usize,
    server_closing: Arc<AtomicBool>,
    stats: Arc<Stats>,
}

/// The producer half of one streaming response.
///
/// Cheap to clone; all clones feed the same stream. Every method is
/// non-async and safe to call from any thread. Only [`StreamSink::push`],
/// [`StreamSink::keepalive`], [`StreamSink::push_many`], [`StreamSink::done`]
/// and [`StreamSink::wait_send`] may block, and only while the channel is full.
#[derive(Clone)]
pub struct StreamSink {
    shared: Arc<SinkShared>,
}

/// Create a sink, the receiver of its head, and the body that streams it.
pub(crate) fn channel(
    max_bytes: usize,
    server_closing: Arc<AtomicBool>,
    stats: Arc<Stats>,
    flusher: Arc<Flusher>,
) -> (StreamSink, oneshot::Receiver<Head>, SseBody) {
    let (head_tx, head_rx) = oneshot::channel();
    let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
    let shared = Arc::new(SinkShared {
        head: Mutex::new(Some(head_tx)),
        tx: Mutex::new(Some(tx)),
        waker: Mutex::new(None),
        flusher,
        closed: AtomicBool::new(false),
        aborted: AtomicBool::new(false),
        queued_bytes: AtomicUsize::new(0),
        max_bytes,
        server_closing,
        stats,
    });
    let body = SseBody { rx, shared: shared.clone(), armed: false, ended: false };
    (StreamSink { shared }, head_rx, body)
}

/// Lock a mutex, recovering the data if a panicking thread poisoned it (the
/// guarded values stay consistent: they are only ever taken or read).
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl StreamSink {
    /// Answer with a plain JSON response instead of a stream. Only valid
    /// before [`StreamSink::start`]; returns `false` if the head was already
    /// decided (or the request is gone).
    pub fn reply(&self, status: u16, body: impl Into<Bytes>) -> bool {
        let sent = self.send_head(Head::Reply { status, body: body.into() });
        // No event can follow a JSON reply.
        self.shared.closed.store(true, Ordering::Release);
        sent
    }

    /// Send the 200 event-stream head. Returns `false` if the head was
    /// already decided or the request is gone.
    pub fn start(&self) -> bool {
        self.send_head(Head::Start)
    }

    fn send_head(&self, head: Head) -> bool {
        match lock(&self.shared.head).take() {
            Some(tx) => tx.send(head).is_ok(),
            None => false,
        }
    }

    /// Whether producing more is pointless: the client is gone, the stream
    /// was closed, or the server is shutting down.
    pub fn is_closed(&self) -> bool {
        self.shared.closed.load(Ordering::Acquire) || self.shared.server_closing.load(Ordering::Acquire)
    }

    /// Queue one `data: <payload>\n\n` event, without blocking.
    pub fn try_push(&self, payload: &[u8]) -> SendOutcome {
        let mut frame = Vec::new();
        format_data_event(payload, &mut frame);
        self.try_send(Bytes::from(frame), Kind::Data(1))
    }

    /// Queue several events as one channel item, without blocking.
    pub fn try_push_many<'a, I>(&self, payloads: I) -> SendOutcome
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut frame = Vec::new();
        let mut count = 0u64;
        for payload in payloads {
            format_data_event(payload, &mut frame);
            count += 1;
        }
        if count == 0 {
            return if self.is_closed() { SendOutcome::Closed } else { SendOutcome::Sent };
        }
        self.try_send(Bytes::from(frame), Kind::Data(count))
    }

    /// Queue the `: keepalive` comment, without blocking. Same closed and
    /// backpressure rules as [`StreamSink::try_push`].
    pub fn try_keepalive(&self) -> SendOutcome {
        self.try_send(Bytes::from_static(KEEPALIVE_EVENT), Kind::Keepalive)
    }

    /// Queue `data: [DONE]`, without blocking. Call [`StreamSink::finish`]
    /// after it has been sent.
    pub fn try_done(&self) -> SendOutcome {
        self.try_send(Bytes::from_static(DONE_EVENT), Kind::Done)
    }

    /// Finish a send that found the channel full, waiting for room. Returns
    /// whether the frame was queued.
    ///
    /// Blocks the calling thread; never call it from an async context.
    pub fn wait_send(&self, pending: PendingFrame) -> bool {
        let len = pending.frame.len();
        let closed = self.shared.closed.load(Ordering::Acquire)
            || (!pending.done && self.shared.server_closing.load(Ordering::Acquire));
        let tx = match lock(&self.shared.tx).as_ref() {
            Some(tx) if !closed => tx.clone(),
            _ => return self.unqueue(len),
        };
        match tx.blocking_send(pending.frame) {
            Ok(()) => {
                self.wake_body();
                true
            }
            Err(_) => {
                self.shared.closed.store(true, Ordering::Release);
                self.unqueue(len)
            }
        }
    }

    /// Queue one event, waiting while the channel is full. `false` once the
    /// stream is closed.
    pub fn push(&self, payload: &[u8]) -> bool {
        self.complete(self.try_push(payload))
    }

    /// Queue several events at once, waiting while the channel is full.
    pub fn push_many<'a, I>(&self, payloads: I) -> bool
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        self.complete(self.try_push_many(payloads))
    }

    /// Queue the keepalive comment, waiting while the channel is full.
    pub fn keepalive(&self) -> bool {
        self.complete(self.try_keepalive())
    }

    /// Queue `data: [DONE]` and end the stream.
    pub fn done(&self) {
        let outcome = self.try_done();
        self.complete(outcome);
        self.finish();
    }

    /// End the stream: no more events can be queued, and the body ends after
    /// what is already queued. Also drops an undecided head, which makes the
    /// transport answer 500. Idempotent.
    pub fn finish(&self) {
        lock(&self.shared.head).take();
        if lock(&self.shared.tx).take().is_some() {
            // The body learns that the channel ended on its next poll.
            self.wake_body();
        }
    }

    /// Hand the body's waker (if it is waiting) to the flusher.
    fn wake_body(&self) {
        let waker = lock(&self.shared.waker).take();
        if let Some(waker) = waker {
            self.shared.flusher.wake(waker);
        }
    }

    fn complete(&self, outcome: SendOutcome) -> bool {
        match outcome {
            SendOutcome::Sent => true,
            SendOutcome::Full(pending) => self.wait_send(pending),
            SendOutcome::Closed => false,
        }
    }

    fn try_send(&self, frame: Bytes, kind: Kind) -> SendOutcome {
        // The terminal event is still delivered while the server drains.
        let closing = self.shared.server_closing.load(Ordering::Acquire) && !matches!(kind, Kind::Done);
        if closing || self.shared.closed.load(Ordering::Acquire) {
            return SendOutcome::Closed;
        }
        let len = frame.len();
        let before = self.shared.queued_bytes.fetch_add(len, Ordering::AcqRel);
        // A single event larger than the bound is allowed onto an empty queue:
        // the bound is about a client falling behind, not about event size.
        if before > 0 && before + len > self.shared.max_bytes {
            self.shared.queued_bytes.fetch_sub(len, Ordering::AcqRel);
            if !self.shared.aborted.swap(true, Ordering::AcqRel) {
                stats::inc(&self.shared.stats.slow_client_closes);
            }
            self.shared.closed.store(true, Ordering::Release);
            // The body aborts the connection on its next poll.
            self.wake_body();
            return SendOutcome::Closed;
        }
        let result = match lock(&self.shared.tx).as_ref() {
            Some(tx) => tx.try_send(frame),
            None => {
                self.unqueue(len);
                return SendOutcome::Closed;
            }
        };
        match result {
            Ok(()) => {
                kind.count(&self.shared.stats);
                self.wake_body();
                SendOutcome::Sent
            }
            Err(mpsc::error::TrySendError::Full(frame)) => {
                kind.count(&self.shared.stats);
                SendOutcome::Full(PendingFrame { frame, done: matches!(kind, Kind::Done) })
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.shared.closed.store(true, Ordering::Release);
                self.unqueue(len);
                SendOutcome::Closed
            }
        }
    }

    /// Take back the byte count of a frame that was not queued; returns
    /// `false` for use as a send result.
    fn unqueue(&self, len: usize) -> bool {
        self.shared.queued_bytes.fetch_sub(len, Ordering::AcqRel);
        false
    }

    /// Bytes queued and not yet taken by the body (for tests and stats).
    pub fn queued_bytes(&self) -> usize {
        self.shared.queued_bytes.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Data(u64),
    Keepalive,
    Done,
}

impl Kind {
    fn count(self, stats: &Stats) {
        match self {
            Kind::Data(n) => stats::add(&stats.chunks_pushed, n),
            Kind::Keepalive => stats::inc(&stats.keepalives_pushed),
            Kind::Done => {}
        }
    }
}

/// The response body of an event stream.
pub(crate) struct SseBody {
    rx: mpsc::Receiver<Bytes>,
    shared: Arc<SinkShared>,
    /// Set once the body is used as a response (counts as an open stream).
    armed: bool,
    /// The channel was drained to its end (the producer finished).
    ended: bool,
}

impl SseBody {
    /// Mark this body as a live response: counted in `open_streams`, and in
    /// `cancelled` if it is dropped before it ends.
    pub fn arm(&mut self) {
        if !self.armed {
            self.armed = true;
            stats::inc(&self.shared.stats.open_streams);
        }
    }

    /// One frame from `first` and whatever else is queued, accounted.
    fn emit(&mut self, first: Bytes) -> Frame<Bytes> {
        let frame = self.coalesce(first);
        let len = frame.len();
        self.shared.queued_bytes.fetch_sub(len, Ordering::AcqRel);
        stats::inc(&self.shared.stats.frames_written);
        stats::add(&self.shared.stats.bytes_written, len as u64);
        Frame::data(frame)
    }

    fn end(&mut self) -> Poll<Option<Result<Frame<Bytes>, io::Error>>> {
        self.ended = true;
        Poll::Ready(None)
    }

    /// Gather everything queued, starting with `first`, into one frame.
    fn coalesce(&mut self, first: Bytes) -> Bytes {
        if first.len() >= COALESCE_CAP {
            return first;
        }
        let second = match self.rx.try_recv() {
            Ok(next) => next,
            Err(_) => return first,
        };
        let mut buf = BytesMut::with_capacity((first.len() + second.len()).max(4096));
        buf.extend_from_slice(&first);
        buf.extend_from_slice(&second);
        while buf.len() < COALESCE_CAP {
            match self.rx.try_recv() {
                Ok(next) => buf.extend_from_slice(&next),
                Err(_) => break,
            }
        }
        buf.freeze()
    }
}

impl Body for SseBody {
    type Data = Bytes;
    type Error = io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, io::Error>>> {
        if self.shared.aborted.load(Ordering::Acquire) {
            return Poll::Ready(Some(Err(io::Error::other("stream closed: client fell too far behind"))));
        }
        match self.rx.try_recv() {
            Ok(first) => return Poll::Ready(Some(Ok(self.emit(first)))),
            Err(TryRecvError::Disconnected) => return self.end(),
            Err(TryRecvError::Empty) => {}
        }
        *lock(&self.shared.waker) = Some(cx.waker().clone());
        // A producer that queued between the try_recv above and storing the
        // waker found no waker to hand over, so look again.
        match self.rx.try_recv() {
            Ok(first) => {
                lock(&self.shared.waker).take();
                Poll::Ready(Some(Ok(self.emit(first))))
            }
            Err(TryRecvError::Disconnected) => {
                lock(&self.shared.waker).take();
                self.end()
            }
            Err(TryRecvError::Empty) => {
                if self.shared.aborted.load(Ordering::Acquire) {
                    return Poll::Ready(Some(Err(io::Error::other("stream closed: client fell too far behind"))));
                }
                Poll::Pending
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.ended
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

impl Drop for SseBody {
    fn drop(&mut self) {
        // Whatever the reason (finished, client gone, write failed, shutdown
        // abort), producers must stop now.
        self.shared.closed.store(true, Ordering::Release);
        if self.armed {
            stats::dec(&self.shared.stats.open_streams);
            if !self.ended {
                stats::inc(&self.shared.stats.cancelled);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
    use std::time::Duration;

    fn new_channel(max_bytes: usize) -> (StreamSink, oneshot::Receiver<Head>, SseBody, Arc<Stats>) {
        let stats = Arc::new(Stats::default());
        let (sink, head, body) = channel(max_bytes, Arc::new(AtomicBool::new(false)), stats.clone(), Flusher::direct());
        (sink, head, body, stats)
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap()
    }

    #[test]
    fn slow_consumer_gets_coalesced_frames_with_identical_bytes() {
        let (sink, _head, mut body, stats) = new_channel(8 << 20);
        let producer = std::thread::spawn(move || {
            let mut expected = Vec::new();
            for i in 0..1000 {
                let payload = format!("{{\"n\":{i}}}");
                assert!(sink.push(payload.as_bytes()));
                format_data_event(payload.as_bytes(), &mut expected);
                if i % 10 == 0 {
                    assert!(sink.keepalive());
                    expected.extend_from_slice(KEEPALIVE_EVENT);
                }
            }
            sink.done();
            expected.extend_from_slice(DONE_EVENT);
            expected
        });
        let received = runtime().block_on(async {
            let mut received = Vec::new();
            while let Some(frame) = body.frame().await {
                received.extend_from_slice(frame.unwrap().data_ref().unwrap());
                // A slow reader: the producer fills the channel meanwhile.
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            received
        });
        let expected = producer.join().unwrap();
        assert_eq!(received, expected);
        let frames = stats.frames_written.load(Ordering::Relaxed);
        let events = 1000 + 100 + 1;
        assert!(frames * 20 < events, "{frames} frames for {events} events");
        assert_eq!(stats.chunks_pushed.load(Ordering::Relaxed), 1000);
        assert_eq!(stats.keepalives_pushed.load(Ordering::Relaxed), 100);
        assert_eq!(sink_queued(&body), 0);
    }

    fn sink_queued(body: &SseBody) -> usize {
        body.shared.queued_bytes.load(Ordering::Acquire)
    }

    #[test]
    fn dropping_the_body_closes_the_sink() {
        let (sink, _head, body, _stats) = new_channel(8 << 20);
        assert!(sink.push(b"{}"));
        drop(body);
        assert!(sink.is_closed());
        assert!(!sink.push(b"{}"));
        assert!(!sink.keepalive());
        assert_eq!(sink.queued_bytes(), "data: {}\n\n".len());
    }

    #[test]
    fn full_channel_reports_full_then_wait_send_completes() {
        let (sink, _head, mut body, _stats) = new_channel(8 << 20);
        for _ in 0..CHANNEL_CAPACITY {
            assert!(matches!(sink.try_push(b"1"), SendOutcome::Sent));
        }
        let pending = match sink.try_push(b"2") {
            SendOutcome::Full(pending) => pending,
            other => panic!("expected Full, got {other:?}"),
        };
        let waiter = {
            let sink = sink.clone();
            std::thread::spawn(move || sink.wait_send(pending))
        };
        runtime().block_on(async {
            let frame = body.frame().await.unwrap().unwrap();
            assert!(!frame.data_ref().unwrap().is_empty());
        });
        assert!(waiter.join().unwrap());
    }

    #[test]
    fn byte_bound_closes_and_aborts_the_body() {
        let (sink, _head, mut body, stats) = new_channel(1000);
        let big = vec![b'x'; 600];
        assert!(sink.push(&big), "one oversized event onto an empty queue is allowed");
        assert!(!sink.push(&big), "the second exceeds the bound");
        assert!(sink.is_closed());
        assert_eq!(stats.slow_client_closes.load(Ordering::Relaxed), 1);
        runtime().block_on(async {
            assert!(body.frame().await.unwrap().is_err(), "the body aborts the connection");
        });
    }

    #[test]
    fn reply_decides_the_head_once() {
        let (sink, head, _body, _stats) = new_channel(8 << 20);
        assert!(sink.reply(400, &b"{}"[..]));
        assert!(!sink.start());
        assert!(!sink.push(b"{}"), "no events after a JSON reply");
        match runtime().block_on(head).unwrap() {
            Head::Reply { status, body } => {
                assert_eq!(status, 400);
                assert_eq!(&body[..], b"{}");
            }
            Head::Start => panic!("expected a reply"),
        }
    }

    #[test]
    fn finish_without_head_drops_the_head_sender() {
        let (sink, head, _body, _stats) = new_channel(8 << 20);
        sink.finish();
        assert!(runtime().block_on(head).is_err());
    }

    #[test]
    fn server_closing_stops_pushes_but_lets_done_through() {
        let stats = Arc::new(Stats::default());
        let closing = Arc::new(AtomicBool::new(false));
        let (sink, _head, body) = channel(8 << 20, closing.clone(), stats, Flusher::direct());
        assert!(sink.push(b"{}"));
        closing.store(true, Ordering::Release);
        assert!(sink.is_closed());
        assert!(!sink.push(b"{}"));
        assert!(!sink.keepalive());
        sink.done();
        let bytes = runtime().block_on(async { body.collect().await.unwrap().to_bytes() });
        assert_eq!(&bytes[..], b"data: {}\n\ndata: [DONE]\n\n");
    }

    #[test]
    fn done_waits_for_room_during_shutdown() {
        let stats = Arc::new(Stats::default());
        let closing = Arc::new(AtomicBool::new(false));
        let (sink, _head, body) = channel(8 << 20, closing.clone(), stats, Flusher::direct());
        for _ in 0..CHANNEL_CAPACITY {
            assert!(matches!(sink.try_push(b"1"), SendOutcome::Sent));
        }
        closing.store(true, Ordering::Release);
        let producer = std::thread::spawn(move || sink.done());
        let bytes = runtime().block_on(async {
            // Drain slowly, so done() finds the channel full and has to wait.
            tokio::time::sleep(Duration::from_millis(50)).await;
            body.collect().await.unwrap().to_bytes()
        });
        producer.join().unwrap();
        assert!(bytes.ends_with(DONE_EVENT), "the [DONE] event was dropped");
        assert_eq!(bytes.len(), CHANNEL_CAPACITY * b"data: 1\n\n".len() + DONE_EVENT.len());
    }

    fn ticked_runtime(tick: Duration) -> (tokio::runtime::Runtime, Arc<Flusher>, Arc<Stats>) {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let stats = Arc::new(Stats::default());
        let flusher = Flusher::new(tick, stats.clone());
        rt.spawn(flusher.clone().run());
        (rt, flusher, stats)
    }

    #[test]
    fn batched_wakes_deliver_every_event_of_many_streams() {
        let (rt, flusher, stats) = ticked_runtime(Duration::from_millis(1));
        let closing = Arc::new(AtomicBool::new(false));
        let mut producers = Vec::new();
        let mut readers = Vec::new();
        for s in 0..40u64 {
            let (sink, _head, body) = channel(8 << 20, closing.clone(), stats.clone(), flusher.clone());
            readers.push(rt.spawn(async move { body.collect().await.unwrap().to_bytes() }));
            producers.push(std::thread::spawn(move || {
                let mut expected = Vec::new();
                for i in 0..3000u64 {
                    let payload = format!("{{\"s\":{s},\"n\":{i}}}");
                    assert!(sink.push(payload.as_bytes()));
                    format_data_event(payload.as_bytes(), &mut expected);
                    // Jitter, so pushes land both while the body is parked and
                    // while a wake for it is pending.
                    if (i + s) % 7 == 0 {
                        std::thread::sleep(Duration::from_micros((i * 37 + s) % 50));
                    }
                }
                sink.done();
                expected.extend_from_slice(DONE_EVENT);
                expected
            }));
        }
        for (producer, reader) in producers.into_iter().zip(readers) {
            let expected = producer.join().unwrap();
            let received = rt.block_on(async { tokio::time::timeout(Duration::from_secs(10), reader).await });
            let received = received.expect("a stream never got its last wake").unwrap();
            assert_eq!(&received[..], &expected[..]);
        }
        let batches = stats.flush_batches.load(Ordering::Relaxed);
        let wakes = stats.flush_wakes.load(Ordering::Relaxed);
        assert!(batches > 0 && wakes >= batches, "{batches} batches, {wakes} wakes");
    }

    #[test]
    fn each_push_is_delivered_without_waiting_for_the_next() {
        // Ping-pong: every push waits until its reader saw it, so a wake lost
        // between the body's empty check and storing its waker shows up as a
        // stuck stream (nothing else would wake it), not as a late event.
        for tick in [Duration::ZERO, Duration::from_micros(200)] {
            let (rt, flusher, stats) = ticked_runtime(tick);
            let closing = Arc::new(AtomicBool::new(false));
            let mut producers = Vec::new();
            for _ in 0..8 {
                let (sink, _head, mut body) = channel(8 << 20, closing.clone(), stats.clone(), flusher.clone());
                let (ack_tx, ack_rx) = std::sync::mpsc::channel::<()>();
                rt.spawn(async move {
                    while let Some(frame) = body.frame().await {
                        frame.unwrap();
                        let _ = ack_tx.send(());
                    }
                });
                producers.push(std::thread::spawn(move || {
                    for i in 0..1500u32 {
                        assert!(sink.push(b"{}"));
                        if let Err(err) = ack_rx.recv_timeout(Duration::from_secs(2)) {
                            panic!("push {i} (tick {tick:?}) was never delivered: {err}");
                        }
                    }
                    sink.done();
                }));
            }
            for producer in producers {
                producer.join().unwrap();
            }
        }
    }

    #[test]
    fn a_push_to_an_idle_server_is_not_held_for_a_tick() {
        // A long tick makes any wait for it obvious.
        let (rt, flusher, stats) = ticked_runtime(Duration::from_millis(300));
        let (sink, _head, mut body) = channel(8 << 20, Arc::new(AtomicBool::new(false)), stats, flusher);
        let reader = rt.spawn(async move {
            let started = std::time::Instant::now();
            let frame = body.frame().await.unwrap().unwrap();
            (started.elapsed(), frame.into_data().unwrap())
        });
        std::thread::sleep(Duration::from_millis(50)); // the body is parked now
        let pushed = std::time::Instant::now();
        assert!(sink.push(b"{}"));
        let (_, data) = rt.block_on(reader).unwrap();
        assert_eq!(&data[..], b"data: {}\n\n");
        let latency = pushed.elapsed();
        assert!(latency < Duration::from_millis(100), "the first wake waited {latency:?}");
    }

    #[test]
    fn push_many_frames_every_payload() {
        let (sink, _head, body, stats) = new_channel(8 << 20);
        assert!(sink.push_many([&b"{\"a\":1}"[..], b"{\"b\":2}"]));
        sink.done();
        let bytes = runtime().block_on(async { body.collect().await.unwrap().to_bytes() });
        assert_eq!(&bytes[..], b"data: {\"a\":1}\n\ndata: {\"b\":2}\n\ndata: [DONE]\n\n");
        assert_eq!(stats.chunks_pushed.load(Ordering::Relaxed), 2);
    }
}
