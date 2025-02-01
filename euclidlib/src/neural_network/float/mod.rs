use std::ops::*;

pub mod f32;
pub mod f64;
pub mod qf8;

pub use qf8::*;

/// Generic float numbers representation.
/// This can be used to optimize floats size for your neurons.
///
/// It's generally recommended to implement both positive and negative
/// sides, at least from -1.0 to 1.0, because this float type will be used
/// everywhere within the neural network module, including gradients calculations
/// for backward propagation, and with too little float range these calculations
/// could be incorrect and produce mistakes during the network training.
pub trait Float:
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> +
    AddAssign + SubAssign + MulAssign + DivAssign + Neg<Output = Self> +
    Default + Clone + Copy + PartialEq + std::fmt::Debug + std::fmt::Display + Sized
{
    /// Constant equal to `Self::from_float(0.0)`.
    /// Needed for compiler optimizations.
    ///
    /// Use `-Float::ZERO` for negative zero.
    const ZERO: Self;

    /// Constant equal to `Self::from_float(0.5)`.
    /// Needed for compiler optimizations.
    const HALF: Self;

    /// Constant equal to `Self::from_float(1.0)`.
    /// Needed for compiler optimizations.
    const ONE: Self;

    /// Minimal allowed value of the float.
    const MIN: Self;

    /// Maximal allowed value of the float.
    const MAX: Self;

    /// Machine epsilon value.
    ///
    /// This is the difference between 1.0 and the next
    /// larger representable number.
    const EPSILON: Self;

    /// Size of the float in bytes.
    const BYTES: usize;

    /// Represent current float as machine f32.
    fn as_f32(&self) -> f32;

    /// Represent current float as machine f64 (double precision).
    fn as_f64(&self) -> f64;

    /// Convert given float to another type.
    fn from_float<F: Float>(float: F) -> Self;

    /// Convert current float to bytes representation.
    fn to_bytes(&self) -> [u8; Self::BYTES];

    /// Decode float from bytes representation.
    fn from_bytes(bytes: &[u8; Self::BYTES]) -> Self;

    /// Check precision error after storing given float
    /// in the current type.
    ///
    /// Returns `new_float - float` value.
    fn precision_error(float: f64) -> f64 {
        Self::from_float(float).as_f64() - float
    }

    // =================================== Arithmetic functions ===================================

    /// Returns true if self has a positive sign, including +0.0.
    fn is_positive(&self) -> bool {
        self.as_f32().is_sign_positive()
    }

    /// Returns true if self has a negative sign, including -0.0.
    fn is_negative(&self) -> bool {
        self.as_f32().is_sign_negative()
    }

    /// Computes the absolute value of self.
    fn abs(&self) -> Self {
        Self::from_float(self.as_f64().abs())
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    fn log(&self, base: Self) -> Self {
        Self::from_float(self.as_f64().log(base.as_f64()))
    }

    /// Returns the base 2 logarithm of the number.
    fn log2(&self) -> Self {
        Self::from_float(self.as_f64().log2())
    }

    /// Returns the base 10 logarithm of the number.
    fn log10(&self) -> Self {
        Self::from_float(self.as_f64().log10())
    }

    /// Returns the natural logarithm of the number.
    fn ln(&self) -> Self {
        Self::from_float(self.as_f64().ln())
    }

    /// Raises a number to a floating point power.
    fn powf(&self, n: Self) -> Self {
        Self::from_float(self.as_f64().powf(n.as_f64()))
    }

    /// Raises a number to an integer power.
    ///
    /// Using this function is generally faster than using powf. It might have
    /// a different sequence of rounding operations than powf, so the results
    /// are not guaranteed to agree.
    fn powi(&self, n: i32) -> Self {
        Self::from_float(self.as_f64().powi(n))
    }

    /// Calculate `e^(self)`.
    fn exp(&self) -> Self {
        Self::from_float(self.as_f64().exp())
    }

    /// Returns the square root of a number.
    fn sqrt(&self) -> Self {
        Self::from_float(self.as_f64().sqrt())
    }

    /// Computes the sine of a number (in radians).
    fn sin(&self) -> Self {
        Self::from_float(self.as_f64().sin())
    }

    /// Computes the cosine of a number (in radians).
    fn cos(&self) -> Self {
        Self::from_float(self.as_f64().cos())
    }

    /// Computes the tangent of a number (in radians).
    fn tan(&self) -> Self {
        Self::from_float(self.as_f64().tan())
    }

    /// Hyperbolic sine function.
    fn sinh(&self) -> Self {
        Self::from_float(self.as_f64().sinh())
    }

    /// Hyperbolic cosine function.
    fn cosh(&self) -> Self {
        Self::from_float(self.as_f64().cosh())
    }

    /// Hyperbolic tangent function.
    fn tanh(&self) -> Self {
        Self::from_float(self.as_f64().tanh())
    }
}
