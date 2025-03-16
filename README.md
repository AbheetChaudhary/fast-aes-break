# fast-aes-break
Breaking AES encryption using power traces in sub millisecond time.

# TODO
- [ ] Add multithreading, make sure threads are not hyperthread syblings
- [ ] Use SIMD for pearson correlation calculation
- [ ] Experiment to get better cache locality

# Prerequisities
- Any Unix like operating system
- Stable Rust toolchain, see below
- Power trace file in HDF5 format. Rename it to `foobarbaz.h5`, or use the
  default file provided in this repo.

## To Get Stable Rust Toolchain
run `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`, then
choose to proceed with standard installation.

To test if everything was installed correctly, run the following commands:
`cargo --version`, `rustc --version`

## Building
Run `cargo build` to create the debug build, or `cargo build --release` to
create the aggressively optimized version.

## Runing
`cargo run` to run the debug version or `cargo run --release` to run the
release version.
