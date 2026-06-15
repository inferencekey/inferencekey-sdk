//! FFI helpers: string marshalling and the thread-local last-error slot.
//!
//! All raw-pointer handling for the C ABI lives here so the entry points in
//! `lib.rs` read as small, intention-revealing steps.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CStr, CString};

use crate::IK_OK;

thread_local! {
    /// The most recent error message on this thread, taken by `ik_last_error`.
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Record an error message for the calling thread and return a non-zero status.
///
/// Returns `crate::IK_ERR_CALL_FAILED` so callers can `Err(set_last_error(..))`
/// inside a `Result<_, c_int>` pipeline; specific call sites override the code
/// when they need a more precise one.
pub fn set_last_error(message: &str) -> c_int {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(message.to_owned()));
    crate::IK_ERR_CALL_FAILED
}

/// Take (and clear) the calling thread's last error message.
pub fn take_last_error() -> Option<String> {
    LAST_ERROR.with(|slot| slot.borrow_mut().take())
}

/// Borrow a NUL-terminated C string as `&str`, recording an error on null or
/// non-UTF-8 input. `name` names the argument for the error message.
///
/// # Safety
/// `ptr` must be null or a valid NUL-terminated string for the call's duration.
pub unsafe fn borrow_str<'a>(ptr: *const c_char, name: &str) -> Result<&'a str, ()> {
    if ptr.is_null() {
        set_last_error(&format!("{name} must not be null"));
        return Err(());
    }
    CStr::from_ptr(ptr).to_str().map_err(|_| {
        set_last_error(&format!("{name} is not valid UTF-8"));
    })
}

/// Allocate a C string from `s`. Fails only when `s` contains an interior NUL.
pub fn into_c_string(s: &str) -> Result<*mut c_char, ()> {
    CString::new(s).map(CString::into_raw).map_err(|_| ())
}

/// Convert a `Result<(), c_int>` into a C status code (`IK_OK` on success).
pub trait OkOrErr {
    fn ok_or_err(self) -> c_int;
}

impl OkOrErr for Result<(), c_int> {
    fn ok_or_err(self) -> c_int {
        match self {
            Ok(()) => IK_OK,
            Err(code) => code,
        }
    }
}
