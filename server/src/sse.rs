//! Server-sent event streams: the producer half handed to the application
//! ([`StreamSink`]) and the consumer half that becomes the response body
//! ([`SseBody`]).
//!
//! A push frames the event and queues it in a bounded channel: a memcpy, no
//! syscall. The body drains everything already queued (up to
//! [`COALESCE_CAP`]) into one frame on each wakeup, so under load N events
//! cost one socket write instead of N. There is no artificial delay: an event
//! pushed to an idle stream is written at once.

use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};

use bytes::{Bytes, BytesMut};
use hyper::body::{Body, Frame, SizeHint};
use inferencekey_core::domain::sse::{format_data_event, DONE_EVENT, KEEPALIVE_EVENT};
use tokio::sync::{mpsc, oneshot};

use crate::stats::{self, Stats};

/// Queued events per stream before a producer has to wait.
pub const CHANNEL_CAPACITY: usize = 256;

/// The most bytes one coalesced frame gathers (it may exceed this by the last
/// event taken).
pub const COALESCE_CAP: usize = 64 * 1024;

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
) -> (StreamSink, oneshot::Receiver<Head>, SseBody) {
    let (head_tx, head_rx) = oneshot::channel();
    let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
    let shared = Arc::new(SinkShared {
        head: Mutex::new(Some(head_tx)),
        tx: Mutex::new(Some(tx)),
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
            Ok(()) => true,
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
        lock(&self.shared.tx).take();
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
        match self.rx.poll_recv(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                self.ended = true;
                Poll::Ready(None)
            }
            Poll::Ready(Some(first)) => {
                let frame = self.coalesce(first);
                let len = frame.len();
                self.shared.queued_bytes.fetch_sub(len, Ordering::AcqRel);
                stats::inc(&self.shared.stats.frames_written);
                stats::add(&self.shared.stats.bytes_written, len as u64);
                Poll::Ready(Some(Ok(Frame::data(frame))))
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
        let (sink, head, body) = channel(max_bytes, Arc::new(AtomicBool::new(false)), stats.clone());
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
        let (sink, _head, body) = channel(8 << 20, closing.clone(), stats);
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
        let (sink, _head, body) = channel(8 << 20, closing.clone(), stats);
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
