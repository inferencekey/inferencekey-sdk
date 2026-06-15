//! Build hook for the Python extension module.
//!
//! On macOS a Python extension `.dylib` must leave the `_Py*` symbols
//! unresolved at link time — they are provided by the interpreter that loads
//! the module. `maturin` sets this automatically when building wheels; we emit
//! it here too so a plain `cargo build` of the extension also links. (Linux
//! resolves these lazily by default, so no flag is needed there.)
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
