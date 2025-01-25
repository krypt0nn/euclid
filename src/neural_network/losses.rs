use super::prelude::*;

#[inline]
/// `0.5 * (actual - expected)^2`
pub fn quadratic_error<F: Float>(actual: F, expected: F) -> F {
    F::HALF * (actual - expected).powi(2)
}

#[inline]
/// `actual - expected`
pub fn quadratic_error_derivative<F: Float>(actual: F, expected: F) -> F {
    actual - expected
}

#[inline]
/// `-(expected * ln(actual) + (1 - expected) * ln(1 - actual))`
pub fn cross_entropy<F: Float>(actual: F, expected: F) -> F {
    -(expected * actual.ln() + (F::ONE - expected) * (F::ONE - actual).ln())
}

#[inline]
/// `(actual - expected) / (actual * (1 - actual))`
pub fn cross_entropy_derivative<F: Float>(actual: F, expected: F) -> F {
    (actual - expected) / (actual * (F::ONE - actual))
}
