//! Atomic counters served at `GET /debug/stats`.
//!
//! Gauges (`connections`, `open_streams`, `inflight`, `queued`) go up and down;
//! the rest only grow. Every update is one relaxed atomic operation, so the
//! counters cost nothing on the hot path and are read without locks.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default, Debug)]
pub(crate) struct Stats {
    /// Open TCP connections (gauge).
    pub connections: AtomicU64,
    /// Connections accepted since start.
    pub connections_total: AtomicU64,
    /// HTTP requests routed since start.
    pub requests: AtomicU64,
    /// Event streams whose head was sent and whose body is still open (gauge).
    pub open_streams: AtomicU64,
    /// Backend calls running on the blocking pool (gauge).
    pub inflight: AtomicU64,
    /// Requests waiting for an admission permit (gauge).
    pub queued: AtomicU64,
    /// `data:` events pushed by producers.
    pub chunks_pushed: AtomicU64,
    /// Keepalive comments pushed by producers.
    pub keepalives_pushed: AtomicU64,
    /// Body frames handed to hyper (each is one coalesced write).
    pub frames_written: AtomicU64,
    /// Bytes in those frames.
    pub bytes_written: AtomicU64,
    /// Streams whose body was dropped before it ended (client gone, shutdown).
    pub cancelled: AtomicU64,
    /// Streams closed because the client fell `stream_max_bytes` behind.
    pub slow_client_closes: AtomicU64,
    /// `accept()` failures (for example EMFILE).
    pub accept_errors: AtomicU64,
    /// Batches of body wakes the flusher ran (see `sse::Flusher`).
    pub flush_batches: AtomicU64,
    /// Body wakes in those batches.
    pub flush_wakes: AtomicU64,
}

pub(crate) fn inc(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn add(counter: &AtomicU64, n: u64) {
    counter.fetch_add(n, Ordering::Relaxed);
}

pub(crate) fn dec(counter: &AtomicU64) {
    counter.fetch_sub(1, Ordering::Relaxed);
}

impl Stats {
    /// The counters as one compact JSON object.
    pub fn to_json(&self) -> String {
        let fields: [(&str, &AtomicU64); 15] = [
            ("connections", &self.connections),
            ("connections_total", &self.connections_total),
            ("requests", &self.requests),
            ("open_streams", &self.open_streams),
            ("inflight", &self.inflight),
            ("queued", &self.queued),
            ("chunks_pushed", &self.chunks_pushed),
            ("keepalives_pushed", &self.keepalives_pushed),
            ("frames_written", &self.frames_written),
            ("bytes_written", &self.bytes_written),
            ("cancelled", &self.cancelled),
            ("slow_client_closes", &self.slow_client_closes),
            ("accept_errors", &self.accept_errors),
            ("flush_batches", &self.flush_batches),
            ("flush_wakes", &self.flush_wakes),
        ];
        let body: Vec<String> = fields
            .iter()
            .map(|(name, value)| format!("\"{name}\":{}", value.load(Ordering::Relaxed)))
            .collect();
        format!("{{{}}}", body.join(","))
    }
}

/// Decrements a gauge when dropped, so a cancelled future cannot leak a count.
pub(crate) struct GaugeGuard<'a>(&'a AtomicU64);

impl<'a> GaugeGuard<'a> {
    pub fn new(gauge: &'a AtomicU64) -> Self {
        inc(gauge);
        GaugeGuard(gauge)
    }
}

impl Drop for GaugeGuard<'_> {
    fn drop(&mut self) {
        dec(self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_is_valid_and_reflects_counters() {
        let stats = Stats::default();
        inc(&stats.chunks_pushed);
        add(&stats.bytes_written, 42);
        {
            let _g = GaugeGuard::new(&stats.inflight);
            assert_eq!(stats.inflight.load(Ordering::Relaxed), 1);
        }
        let value: serde_json::Value = serde_json::from_str(&stats.to_json()).unwrap();
        assert_eq!(value["chunks_pushed"], 1);
        assert_eq!(value["bytes_written"], 42);
        assert_eq!(value["inflight"], 0);
        assert_eq!(value.as_object().unwrap().len(), 15);
    }
}
