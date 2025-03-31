use core::arch::x86_64::*;

/// Find pearsons coefficient between two vectors of f64 which are of same sizes.
pub fn pearson(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
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

/// Scalar implementation. y is always same in this function, so calculate y_sum,
/// y_sq_sum once and use them repeatedly.
fn pearson_scalar(x: &Vec<f64>, y: &Vec<f64>, y_sum: f64, y_sq_sum: f64) -> f64 {
    let x = AsRef::<[f64]>::as_ref(x);
    let y = AsRef::<[f64]>::as_ref(y);
    let n = x.len();
    assert_eq!(n, y.len());
    let n = n as f64;

    let x_sum: f64 = x.iter().sum();

    let x_sq_sum: f64 = x.iter().map(|_x| _x * _x).sum();

    let prod_sum: f64 = x.iter().zip(y.iter()).map(|(_x, _y)| _x * _y).sum();

    (n * prod_sum - x_sum * y_sum) / 
        f64::sqrt((n * x_sq_sum - x_sum * x_sum) * (n * y_sq_sum - y_sum * y_sum))
}

#[inline]
#[target_feature(enable = "avx2")]
/// Sum 4 f64's in 256-bit wide register
unsafe fn collapse_256(v: __m256d) -> f64 {
    let mut vlow  = _mm256_castpd256_pd128(v);
    let vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128
    let high64 = _mm_unpackhi_pd(vlow, vlow);
    _mm_cvtsd_f64(_mm_add_sd(vlow, high64))  // reduce to scalar
}

/// SIMD version
#[target_feature(enable = "avx2")]
unsafe fn pearson_simd_256(x: &Vec<f64>, y: &Vec<f64>, y_sum: f64, y_sq_sum: f64) -> f64 {
    let n = x.len() as f64;
    let len = x.len();

    let mut x_sum: f64 = 0.0;
    let mut x_sq_sum: f64 = 0.0;
    let mut prod_sum: f64 = 0.0;

    let mut i = 0;

    unsafe {
        let mut x_sum_256 = std::mem::transmute::<[f64; 4], __m256d>([0.0; 4]);
        let mut x_sq_sum_256 = std::mem::transmute::<[f64; 4], __m256d>([0.0; 4]);
        let mut prod_sum_256 = std::mem::transmute::<[f64; 4], __m256d>([0.0; 4]);

        while i + 4 < len {
            let _x = _mm256_loadu_pd(x.get_unchecked(i));
            let _y = _mm256_loadu_pd(y.get_unchecked(i));

            x_sum_256 = _mm256_add_pd(x_sum_256, _x);

            x_sq_sum_256 = _mm256_add_pd(x_sq_sum_256, _mm256_mul_pd(_x, _x));

            prod_sum_256 = _mm256_add_pd(prod_sum_256, _mm256_mul_pd(_x, _y));

            i += 4;
        }

        // collapse sum 4 f64's
        x_sum +=  collapse_256(x_sum_256);
        x_sq_sum +=  collapse_256(x_sq_sum_256);  // reduce to scalar
        prod_sum +=  collapse_256(prod_sum_256);  // reduce to scalar

        while i < len {
            let _x = *x.get_unchecked(i);
            let _y = *y.get_unchecked(i);

            x_sum += _x;

            x_sq_sum += _x * _x;

            prod_sum += _x * _y;

            i += 1;
        }
    }

    // println!("x_sum: {x_sum}, y_sum: {y_sum}, \
    //     x_sq_sum: {x_sq_sum}, y_sq_sum: {y_sq_sum}, prod_sum: {prod_sum}");
    let numerator: f64 = n * prod_sum - x_sum * y_sum;
    let denominator: f64 = f64::sqrt(n * x_sq_sum - x_sum * x_sum) * 
        f64::sqrt(n * y_sq_sum - y_sum * y_sum);

    numerator / denominator
}

/// Conditional simd use.
pub fn pearson_simd(x: &Vec<f64>, y: &Vec<f64>, y_sum: f64, y_sq_sum: f64) -> f64 {
    if std::is_x86_feature_detected!("avx2") {
        return unsafe { pearson_simd_256(x, y, y_sum, y_sq_sum) };
    } else {
        return pearson_scalar(x, y, y_sum, y_sq_sum);
    }
}
