use super::*;

impl Float for f32 {
    const ZERO: Self = 0.0;
    const HALF: Self = 0.5;
    const ONE: Self = 1.0;

    const MIN: Self = f32::MIN;
    const MAX: Self = f32::MAX;
    const EPSILON: Self = f32::EPSILON;

    #[inline]
    fn as_f32(&self) -> f32 {
        *self
    }

    #[inline]
    fn as_f64(&self) -> f64 {
        *self as f64
    }

    #[inline]
    fn from_float<F: Float>(float: F) -> Self {
        float.as_f32()
    }

    fn to_bytes(&self) -> Box<[u8]> {
        Box::new(self.to_be_bytes())
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), 4, "f32 must be stored in 4 bytes");

        f32::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3]
        ])
    }

    #[inline]
    fn precision_error(float: f64) -> f64 {
        (float as f32) as f64 - float
    }

    // =================================== Arithmetic functions ===================================

    #[inline]
    fn is_positive(&self) -> bool {
        f32::is_sign_positive(*self)
    }

    #[inline]
    fn is_negative(&self) -> bool {
        f32::is_sign_negative(*self)
    }

    #[inline]
    fn abs(&self) -> Self {
        f32::abs(*self)
    }

    #[inline]
    fn log(&self, base: Self) -> Self {
        f32::log(*self, base)
    }

    #[inline]
    fn log2(&self) -> Self {
        f32::log2(*self)
    }

    #[inline]
    fn log10(&self) -> Self {
        f32::log10(*self)
    }

    #[inline]
    fn ln(&self) -> Self {
        f32::ln(*self)
    }

    #[inline]
    fn powf(&self, n: Self) -> Self {
        f32::powf(*self, n)
    }

    #[inline]
    fn powi(&self, n: i32) -> Self {
        f32::powi(*self, n)
    }

    #[inline]
    fn exp(&self) -> Self {
        f32::exp(*self)
    }

    #[inline]
    fn sqrt(&self) -> Self {
        f32::sqrt(*self)
    }

    #[inline]
    fn sin(&self) -> Self {
        f32::sin(*self)
    }

    #[inline]
    fn cos(&self) -> Self {
        f32::cos(*self)
    }

    #[inline]
    fn tan(&self) -> Self {
        f32::tan(*self)
    }

    #[inline]
    fn sinh(&self) -> Self {
        f32::sinh(*self)
    }

    #[inline]
    fn cosh(&self) -> Self {
        f32::cosh(*self)
    }

    #[inline]
    fn tanh(&self) -> Self {
        f32::tanh(*self)
    }
}
