//! The listening socket: loopback only, with an explicit backlog.

use std::io;
use std::net::{Ipv4Addr, SocketAddr};

use tokio::net::{TcpListener, TcpSocket};

/// Bind `127.0.0.1:port` (0 picks a free port) and listen with `backlog`.
///
/// `SO_REUSEADDR` matches the stdlib `HTTPServer`, so a restart can re-bind a
/// port left in TIME_WAIT; a port another socket is listening on still fails
/// with `AddrInUse`. The kernel caps `backlog` at `net.core.somaxconn`.
///
/// Must run inside a tokio runtime context (the listener registers with its
/// reactor).
pub(crate) fn bind(port: u16, backlog: u32) -> io::Result<TcpListener> {
    let socket = TcpSocket::new_v4()?;
    socket.set_reuseaddr(true)?;
    socket.bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port)))?;
    socket.listen(backlog)
}
