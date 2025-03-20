# fast-aes-break
Breaking 128-bit AES encryption using power traces within 1 second(~80ms). It
uses 50 encryptions and their power traces with 5000 trace samples.

# TODO
- [x] Add multithreading

  It uses 8 threads. It will be better if they are not sybling threads in
  hyperthreaded environment, but its fine if they are.
- [x] Experiment with SIMD for pearson correlation calculation.

  SIMD performance gains are small due to the size of involved arrays being
  small. Currently it gives about x2.5 improvement.
- [x] Experiment to get better cache locality.

  Current implementation has ~2.13% L1-d cache misses, it is possible to get it
  close to ~0.01% but that implementation involves a lot of write contention and
  is not getting along with the optimizer, so the overall time increases in it.
  (not worthit)
- [x] Do some data clean up to discard useless samples.
  
  All the useful trace samples are clumped together. After finding the first
  correct key guess only consider a fraction of the total samples availavle for 
  further processing. The constant `SURROUND` is used to consider only `SURROUND/2`
  samples around it. Received key may not be correct if this range is too small,
  a runtime error will occur if that happens.
- [ ] Remove hardcoded dependencies on number of threads and key sizes.

  Currently the thread count must be 8 and key size must be 16.

# Prerequisities
- Any Unix like operating system
- Stable Rust toolchain and `cmake`, see below
- Power trace file in HDF5 format. Rename it to `foobarbaz.h5`, or use the
  default file provided in this repo.

## To Get Stable Rust Toolchain
run `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`, then
choose to proceed with standard installation. It should take less than 3-4min 
to install with fast internet connection.

To test if everything was installed correctly, restart your terminal and run the
following commands: `cargo --version`, `rustc --version`, they should succeed.

Later if you want to uninstall rust run `rustup self uninstall`.

Also make sure that you have `cmake` installed.

## Building
Run `cargo build` to create the debug build, or `cargo build --release` to
create the aggressively optimized version.

## Runing
`cargo run` to run the debug version or `cargo run --release` to run the
release version.

## Docs
To create documents(if any) run the following command: `cargo doc --no-deps`.
This will open some documentations in your default browser.
