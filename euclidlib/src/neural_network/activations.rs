use super::prelude::*;

#[inline]
pub fn linear<F: Float>(x: F) -> F {
    x
}

#[inline]
pub fn linear_derivative<F: Float>(_: F) -> F {
    F::ONE
}

// FIXME: derivatives of sigmoid and tanh functions can be optimized

#[inline]
pub fn sigmoid<F: Float>(x: F) -> F {
    let y = x.exp();

    y / (F::ONE + y)
}

#[inline]
pub fn sigmoid_derivative<F: Float>(x: F) -> F {
    let y = sigmoid(x);

    y * (F::ONE - y)
}

#[inline]
pub fn tanh<F: Float>(x: F) -> F {
    x.tanh()
}

#[inline]
pub fn tanh_derivative<F: Float>(x: F) -> F {
    F::ONE - tanh(x).powi(2)
}
