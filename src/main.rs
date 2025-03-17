use ndarray::{Dim, ArrayBase, OwnedRepr, Axis, ArrayView1};
use std::time;
use std::thread;
mod correlation;

/// Consider `SURROUND / 2` samples before and after the center to be useful,
/// discard rest.
const SURROUND: usize = 1500;

/// Number of threads. DO NOT CHANGE it.
const THREADS: usize = 8;

/// HDF5 file path relative to crate root.
const FILEPATH: &'static str = "./foobarbaz.h5";

/// Size of message in bytes.
const MESSAGE_SIZE: usize = 16;

/// Size of key in bytes. DO NOT CHANGE it.
const KEY_SIZE: usize = 16;

/// Number of traces available.
const TRACE_COUNT: usize = 50;

/// Number of samples in each trace.
const TRACE_SAMPLES: usize = 5250;

/// Correct number of samples in each trace after the noise is removed.
const CORRECT_SAMPLES: usize = 5000;

/// AES SBOX array.
const SBOX: [u8; 256] = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16 
];

fn main() -> hdf5::Result<()> {
    // Open hdf5 file.
    let file = hdf5::File::open(FILEPATH)?;

    let trace_array_dataset = file.dataset("trace_array")?;
    let trace_array = trace_array_dataset.read::<f64, Dim<[usize; 2]>>().unwrap();
    assert_eq!(trace_array.shape(), [TRACE_COUNT, TRACE_SAMPLES]);

    let textin_array_dataset = file.dataset("textin_array")?;
    let textin_array = textin_array_dataset.read::<u8, Dim<[usize; 2]>>().unwrap();
    assert_eq!(textin_array.shape(), [TRACE_COUNT, MESSAGE_SIZE]);

    let key_array_dataset = file.dataset("key_array")?;
    let key_array = key_array_dataset.read::<u8, Dim<[usize; 2]>>().unwrap();
    assert_eq!(key_array.shape(), [TRACE_COUNT, KEY_SIZE]);

    // Extract the original key. Assuming each row in key_array is same.
    let key_original = key_array.index_axis(Axis(0), 0).iter()
        .map(|x| *x)
        .collect::<Vec<u8>>();
    println!("original: {key_original:?}");

    // Filter the traces.
    // Create empty array of the same size as the corrected trace_array 2D
    // array. Filter the trace_array and fill the elements in this array one
    // trace at a time.
    let mut trace_array_filtered: ndarray::ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
        ArrayBase::zeros([TRACE_COUNT, CORRECT_SAMPLES]);
    for (row, trace) in trace_array.axis_iter(Axis(0)).enumerate() {
        assert_eq!(trace.shape(), [TRACE_SAMPLES]);
        let real_begin = real_index(trace);
        assert_eq!(trace.iter().skip(real_begin).take(CORRECT_SAMPLES).count(), 
            CORRECT_SAMPLES, "Number of samples after real begin is less than {}", 
            CORRECT_SAMPLES);
        for (col, v) in trace.iter().skip(real_begin).take(CORRECT_SAMPLES).enumerate() {
            *trace_array_filtered.get_mut([row, col]).unwrap() = *v;
        }
    }

    let trace_array = trace_array_filtered;

    // A column of the trace_array 2D matrix. A row is entire trace, a column is 
    // vector of i'th value of each trace.
    let mut trace_columns: Vec<Vec<f64>> = Vec::with_capacity(CORRECT_SAMPLES);
    for col in 0..CORRECT_SAMPLES {
        let mut inner = Vec::with_capacity(TRACE_COUNT); // a column
        inner.extend(trace_array.index_axis(Axis(1), col).iter());
        assert_eq!(inner.len(), TRACE_COUNT);
        trace_columns.push(inner);
    }

    // Each vector inside this vector is 'msg[i] for all msg in input_messages'
    let mut input_columns: Vec<Vec<u8>> = Vec::with_capacity(MESSAGE_SIZE);
    for col in 0..MESSAGE_SIZE {
        let mut inner = Vec::with_capacity(TRACE_COUNT);
        inner.extend(textin_array.index_axis(Axis(1), col).iter());
        assert_eq!(inner.len(), TRACE_COUNT);
        input_columns.push(inner);
    }

    // Find center of the trace samples. 
    // All the samples are not equally useful. Once the encryption starts then
    // all trace samples that give high correlation upon correct key_guess are
    // clumped together. We just need to find the correct trace sample for the
    // first key byte, then for further processing we should only consider trace
    // samples around that.
    let center = {
        let mut max_coeff = 0.0;
        let mut col_idx = 0;
        for key_guess in 0..=255u8 {
            let powers = (&input_columns[0]).into_iter().map(|i: &u8| {
                let idx = key_guess ^ *i;
                power_model(SBOX[idx as usize]) as f64
            }).collect::<Vec<f64>>();

            for (i, trace) in trace_columns.iter().enumerate() {
                let coeff = correlation::pearson(&powers, trace).abs();
                if coeff > max_coeff {
                    max_coeff = coeff;
                    col_idx = i;
                }
            }
        }
        col_idx
    };
    // println!("center: {}", center);

    let first_trace = center.saturating_sub(SURROUND / 2);
    let last_trace = if center + SURROUND / 2 > CORRECT_SAMPLES - 1 {
        CORRECT_SAMPLES - 1
    } else {
        center + SURROUND / 2
    };
    let useful_traces = &trace_columns[first_trace..last_trace];

    let begin = time::Instant::now();

    // Key will be stored here.
    let mut key = Vec::with_capacity(KEY_SIZE);

    // cache preformance:
    // > perf stat -e L1-dcache-loads,L1-dcache-load-misses ./target/release/fast-aes-break
    // 6,435,821,080      L1-dcache-loads:u
    //   138,010,582      L1-dcache-load-misses:u          #    2.14% of all L1-dcache accesses
    let key_solver = |idx_key: usize| {
        let mut max_coeff = 0.0;
        let mut key_at_max = 0;
        for key_guess in 0..=255u8 {
            let powers = (&input_columns[idx_key]).into_iter().map(|i: &u8| {
                let idx = key_guess ^ *i;
                power_model(SBOX[idx as usize]) as f64
            }).collect::<Vec<f64>>();

            for trace in useful_traces {
                let coeff = correlation::pearson(&powers, trace).abs();
                if coeff > max_coeff {
                    max_coeff = coeff;
                    key_at_max = key_guess;
                }
            }
        }
        // println!("trace sample index: {}, key_byte: {}", col_idx, idx_key);
        key_at_max
    };

    // Cache friendly version of key_solver.
    // This one is cache friendly but it involves writes in the 'powers' vector
    // which are difficult for the optimizer to deal with. The previous version
    // did not fit entirely in L1 cache but there were no writes involved.
    //
    // cache performance:
    // > perf stat -e L1-dcache-loads,L1-dcache-load-misses ./target/release/fast-aes-break
    // 15,716,511,519      L1-dcache-loads:u
    //        687,935      L1-dcache-load-misses:u          #    0.00% of all L1-dcache accesses
    let _key_solver_cache = |idx_key: usize| {
        let mut max_coeff = 0.0;
        let mut key_at_max = 0;
        for trace in useful_traces {
            let mut powers = vec![0.0; TRACE_COUNT];

            for key_guess in 0..=255 {
                for (i, p) in powers.iter_mut().enumerate() {
                    let xorred = &input_columns[idx_key][i] ^ key_guess;
                    *p = power_model(SBOX[xorred as usize]) as f64;
                }
                let coeff = correlation::pearson(&powers, trace).abs();
                if coeff > max_coeff {
                    max_coeff = coeff;
                    key_at_max = key_guess;
                }
            }
        }

        key_at_max
    };


    // Number of chunks in which processing will be done. Each chunk will
    // be processed by THREADS number of threads at a time.
    const CHUNKS: usize = if KEY_SIZE % THREADS == 0 {
        KEY_SIZE / THREADS
    } else {
        (KEY_SIZE / THREADS) + 1
    };

    for chunk in 0..CHUNKS {
        let i = chunk * THREADS;
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(THREADS);

            handles.push(s.spawn(|| key_solver(i)));
            handles.push(s.spawn(|| key_solver(i + 1)));
            handles.push(s.spawn(|| key_solver(i + 2)));
            handles.push(s.spawn(|| key_solver(i + 3)));
            handles.push(s.spawn(|| key_solver(i + 4)));
            handles.push(s.spawn(|| key_solver(i + 5)));
            handles.push(s.spawn(|| key_solver(i + 6)));
            handles.push(s.spawn(|| key_solver(i + 7)));

            key.extend(handles.into_iter().map(|h| h.join().unwrap()));
        });
    }
    
    let elapsed = begin.elapsed();
 
    println!("recovered: {:?}", key);

    assert!(key_original.eq(&key), "Recovered and original keys did not matched. Try \
        increasing the surrounding value. Currently SURROUND = {SURROUND}");

    println!("recovered and original keys matched.");
    println!("time: {elapsed:.03?}");

    Ok(())
}

/// Hamming weights of each byte.
const HAMMING_WEIGHTS: [usize; 256] = [
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,
];

/// It maps a byte(output after sbox) to its hamming weight. The assumption 
/// behind this power model is that the more the number of 1 bits, the more
/// power is required to process that byte.
fn power_model(sboxed: u8) -> usize {
    HAMMING_WEIGHTS[sboxed as usize]
}

/// Size of the noise implanted between indices 0-250
const THRESHOLD: usize = 50;

/// Find the real index from which the samples begin. Real index is where an
/// array of ~THRESHOLD equal elements ends.
fn real_index(view: ArrayView1<f64>) -> usize {
    let mut real: Option<usize> = None;
    let mut confidence = 1;
    let mut previous: Option<f64> = None;
    for (i, x) in view.iter().take(250).enumerate() {
        match previous.take() {
            Some(p) if *x == p => {
                confidence += 1;
                if confidence == THRESHOLD {
                    real = Some(i);
                    break;
                }
            }
            Some(_) => {
                confidence = 1;
            }
            None => {}
        }

        previous = Some(*x);
    }

    real.expect("there was supposed to be some real index")
}
