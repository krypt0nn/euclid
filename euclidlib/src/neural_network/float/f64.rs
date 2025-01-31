use super::*;

impl Float for f64 {
    const ZERO: Self = 0.0;
    const HALF: Self = 0.5;
    const ONE: Self = 1.0;

    const MIN: Self = f64::MIN;
    const MAX: Self = f64::MAX;
    const EPSILON: Self = f64::EPSILON;

    const BYTES: usize = 8;

    #[inline]
    fn as_f32(&self) -> f32 {
        *self as f32
    }

    #[inline]
    fn as_f64(&self) -> f64 {
        *self
    }

    #[inline]
    fn from_float<F: Float>(float: F) -> Self {
        float.as_f64()
    }

    #[inline]
    fn to_bytes(&self) -> [u8; Self::BYTES] {
        self.to_be_bytes()
    }

    #[inline]
    fn from_bytes(bytes: &[u8; Self::BYTES]) -> Self {
        f64::from_be_bytes(*bytes)
    }

    #[inline]
    fn precision_error(_: f64) -> f64 {
        0.0
    }

    // =================================== Arithmetic functions ===================================

    #[inline]
    fn is_positive(&self) -> bool {
        f64::is_sign_positive(*self)
    }

    #[inline]
    fn is_negative(&self) -> bool {
        f64::is_sign_negative(*self)
    }

    #[inline]
    fn abs(&self) -> Self {
        f64::abs(*self)
    }

    #[inline]
    fn log(&self, base: Self) -> Self {
        f64::log(*self, base)
    }

    #[inline]
    fn log2(&self) -> Self {
        f64::log2(*self)
    }

    #[inline]
    fn log10(&self) -> Self {
        f64::log10(*self)
    }

    #[inline]
    fn ln(&self) -> Self {
        f64::ln(*self)
    }

    #[inline]
    fn powf(&self, n: Self) -> Self {
        f64::powf(*self, n)
    }

    #[inline]
    fn powi(&self, n: i32) -> Self {
        f64::powi(*self, n)
    }

    #[inline]
    fn exp(&self) -> Self {
        f64::exp(*self)
    }

    #[inline]
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }

    #[inline]
    fn sin(&self) -> Self {
        f64::sin(*self)
    }

    #[inline]
    fn cos(&self) -> Self {
        f64::cos(*self)
    }

    #[inline]
    fn tan(&self) -> Self {
        f64::tan(*self)
    }

    #[inline]
    fn sinh(&self) -> Self {
        f64::sinh(*self)
    }

    #[inline]
    fn cosh(&self) -> Self {
        f64::cosh(*self)
    }

    #[inline]
    fn tanh(&self) -> Self {
        f64::tanh(*self)
    }
}
