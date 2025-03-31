# fast-aes-break
Breaking 128-bit AES encryption using power traces within ~~20min~~ ~~4sec~~
~~1sec~~ ~~300ms~~ ~~80ms~~ 40ms. 

Naive python version takes 20min, unacceptable! Rust in release mode brings it
down to 4sec. Multithreading, SIMD(avx2), decent cache locality and some data
cleaning make it even faster.

It uses 50 encryptions and power traces with 5000 samples in each trace.

# TODO
- [x] Add multithreading

  It uses 4 threads. It will be better if they are not sybling threads in
  hyperthreaded environment, but its fine if they are. The gains are not
  noticable with 8 vs 4 threads  on a 4 core hyperthreaded machine.
- [ ] Experiment with better SIMD for pearson correlation.

  Current SIMD performance gains are small due to the size of involved arrays
  being small. There is still a possibility to make better use of SIMD.
- [x] Experiment to get better cache locality.

  Current implementation has ~2.13% L1-d cache misses, it is possible to get it
  close to ~0.01% but that implementation involves a lot of write contention and
  is not getting along with the optimizer, so the overall time increases in it.
  (not worthit)
- [x] Do some data clean up to discard useless samples.
  
  All the useful trace samples are clumped together. After finding the first
  correct key guess only consider a fraction of the total samples availavle for 
  further processing. The constant `SURROUND` is used to consider only `SURROUND/2`
  samples before & after it. Received key may not be correct if this range is too small,
  a runtime error will occur if that happens.

# Prerequisities
- Any Unix like operating system
- x64 machine, optional avx2. Without `avx2` you will see 4x slowdown.
- Stable Rust toolchain and `cmake`, see below
- Power trace file in HDF5 format. Update its name in `src/main.rs` or use the
  default file provided in this repo. The `trace_default.h5` has three datasets,
  'trace_array'(50 rows, 5000 columns, f64), 'textin_array'(50 messages, u8) and
  'textout_array'(50 messages, u8). Since I already know the key, so I have used
  that to verify the correctness of the recovered key, this makes
  'textout_array' useless.

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
