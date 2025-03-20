/// Find pearsons coefficient between two slices of f64 which are of same sizes.
pub fn _pearson<T: AsRef<[f64]>>(x: &T, y: &T) -> f64 {
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
pub fn pearson_simd(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
        let n = x.len() as f64;
    let len = x.len();

    let mut x_sum: f64 = 0.0;
    let mut y_sum: f64 = 0.0;
    let mut x_sq_sum: f64 = 0.0;
    let mut y_sq_sum: f64 = 0.0;
    let mut prod_sum: f64 = 0.0;

    let mut i = 0;

    unsafe {
        let mut x_sum_128 = std::mem::transmute::<[f64; 2], __m128d>([0.0; 2]);
        let mut y_sum_128 = std::mem::transmute::<[f64; 2], __m128d>([0.0; 2]);
        let mut x_sq_sum_128 = std::mem::transmute::<[f64; 2], __m128d>([0.0; 2]);
        let mut y_sq_sum_128 = std::mem::transmute::<[f64; 2], __m128d>([0.0; 2]);
        let mut prod_sum_128 = std::mem::transmute::<[f64; 2], __m128d>([0.0; 2]);

        while i + 2 < len {
            let _x = _mm_loadu_pd(x.get_unchecked(i));
            let _y = _mm_loadu_pd(y.get_unchecked(i));

            x_sum_128 = _mm_add_pd(x_sum_128, _x);
            y_sum_128 = _mm_add_pd(y_sum_128, _y);

            x_sq_sum_128 = _mm_add_pd(x_sq_sum_128, _mm_mul_pd(_x, _x));
            y_sq_sum_128 = _mm_add_pd(y_sq_sum_128, _mm_mul_pd(_y, _y));

            prod_sum_128 = _mm_add_pd(prod_sum_128, _mm_mul_pd(_x, _y));

            i += 2;
        }

        // collapse sum 4 elements
        let temp = _mm_hadd_pd(x_sum_128, x_sum_128);
        x_sum += _mm_cvtsd_f64(temp);

        let temp = _mm_hadd_pd(y_sum_128, y_sum_128);
        y_sum += _mm_cvtsd_f64(temp);

        let temp = _mm_hadd_pd(x_sq_sum_128, x_sq_sum_128);
        x_sq_sum += _mm_cvtsd_f64(temp);

        let temp = _mm_hadd_pd(y_sq_sum_128, y_sq_sum_128);
        y_sq_sum += _mm_cvtsd_f64(temp);

        let temp = _mm_hadd_pd(prod_sum_128, prod_sum_128);
        prod_sum += _mm_cvtsd_f64(temp);

        while i < len {
            let _x = *x.get_unchecked(i);
            let _y = *y.get_unchecked(i);

            x_sum += _x;
            y_sum += _y;

            x_sq_sum += _x * _x;
            y_sq_sum += _y * _y;

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
