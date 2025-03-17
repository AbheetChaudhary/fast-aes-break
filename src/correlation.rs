/// Find pearsons coefficient between two slices of f64 which are of same sizes.
pub fn pearson<T: AsRef<[f64]>>(x: &T, y: &T) -> f64 {
    let x = AsRef::<[f64]>::as_ref(x);
    let y = AsRef::<[f64]>::as_ref(y);
    let n = x.len();
    assert_eq!(n, y.len());
    let n = n as f64;

    let x_sum: f64 = x.iter().sum();
    let y_sum: f64 = y.iter().sum();

    let x_sq_sum: f64 = x.iter().map(|_x| _x * _x).sum();
    let y_sq_sum: f64 = y.iter().map(|_y| _y * _y).sum();

    let prod_sum: f64 = x.iter().zip(y.iter()).map(|(_x, _y)| _x * _y).sum();

    (n * prod_sum - x_sum * y_sum) / 
        f64::sqrt((n * x_sq_sum - x_sum * x_sum) * (n * y_sq_sum - y_sum * y_sum))
}

use core::arch::x86_64::*;

/// SIMD implementation. NOTE: This is not fast enough. Probably the overhead of
/// simd is shadowing its benefits in arrays of size ~50.
pub fn _pearson_simd<T: AsRef<[f64]>>(x: &T, y: &T) -> f64 {
    let x = AsRef::<[f64]>::as_ref(x);
    let y = AsRef::<[f64]>::as_ref(y);
    let n = x.len();
    assert_eq!(n, y.len());
    let n = n as f64;

    // Initialize accumulators
    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    let mut x_sq_sum = 0.0;
    let mut y_sq_sum = 0.0;
    let mut prod_sum = 0.0;

    let mut i = 0;
    unsafe {
        // Process in chunks of 2 using SSE
        while i + 2 <= n as usize {
            let x_vec = _mm_loadu_pd(x[i..].as_ptr());
            let y_vec = _mm_loadu_pd(y[i..].as_ptr());

            let x_sum_vec = _mm_hadd_pd(x_vec, x_vec);
            let x_sum_vec = _mm_hadd_pd(x_sum_vec, x_sum_vec);
            x_sum += _mm_cvtsd_f64(x_sum_vec);

            let y_sum_vec = _mm_hadd_pd(y_vec, y_vec);
            let y_sum_vec = _mm_hadd_pd(y_sum_vec, y_sum_vec);
            y_sum += _mm_cvtsd_f64(y_sum_vec);

            let x_sq_vec = _mm_mul_pd(x_vec, x_vec);
            let y_sq_vec = _mm_mul_pd(y_vec, y_vec);
            let x_sq_sum_vec = _mm_hadd_pd(x_sq_vec, x_sq_vec);
            let x_sq_sum_vec = _mm_hadd_pd(x_sq_sum_vec, x_sq_sum_vec);
            let y_sq_sum_vec = _mm_hadd_pd(y_sq_vec, y_sq_vec);
            let y_sq_sum_vec = _mm_hadd_pd(y_sq_sum_vec, y_sq_sum_vec);
            x_sq_sum += _mm_cvtsd_f64(x_sq_sum_vec);
            y_sq_sum += _mm_cvtsd_f64(y_sq_sum_vec);

            let prod_vec = _mm_mul_pd(x_vec, y_vec);
            let prod_sum_vec = _mm_hadd_pd(prod_vec, prod_vec);
            let prod_sum_vec = _mm_hadd_pd(prod_sum_vec, prod_sum_vec);
            prod_sum += _mm_cvtsd_f64(prod_sum_vec);

            i += 2;
        }
    }

    // Handle any remaining values.
    for j in x.iter().skip(i) {
        x_sum += *j;
    }
    for j in y.iter().skip(i) {
        y_sum += *j;
    }
    for j in x.iter().skip(i).zip(y.iter().skip(i)) {
        prod_sum += j.0 * j.1;
    }
    for j in x.iter().skip(i) {
        x_sq_sum += j * j;
    }
    for j in y.iter().skip(i) {
        y_sq_sum += j * j;
    }

    let numerator = n * prod_sum - x_sum * y_sum;
    let denominator = (n * x_sq_sum - x_sum * x_sum) * (n * y_sq_sum - y_sum * y_sum);
    numerator / f64::sqrt(denominator)
}
