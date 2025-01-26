//! Generic 1-byte float number with a sign bit, 2 exponent bits and 5 mantissa bits.
//!
//! This mod provides aliases to pre-defined generic structs with different
//! parameters. Their name scheme follows this rule:
//!
//! - `qf8_n`   - quantized 1-byte float with `(-n, +n)` values range.
//! - `qf8_n_v` - quantized 1-byte float with `(-n, +n)` values range
//!               and specific distribution rules.
//!
//! `qf8_n_1` generally have larger uniformal distribution around zero
//! with first two ranges united together, while other variants have
//! different ranges with different distribution rules.
//!
//! Examples:
//!
//! - `qf8_2` - uniformal distribution on `(-2, 2)` range.
//! - `qf8_2_1` - different distribution rules within `(-2, 2)` values space.

use super::*;

mod qf8_1_type;

mod qf8_2_type;
mod qf8_2_1_type;

mod qf8_4_type;
mod qf8_4_1_type;
mod qf8_4_2_type;

mod qf8_8_type;
mod qf8_8_1_type;
mod qf8_8_2_type;

mod qf8_16_type;
mod qf8_16_1_type;
mod qf8_16_2_type;

pub use qf8_1_type::qf8_1;

pub use qf8_2_type::qf8_2;
pub use qf8_2_1_type::qf8_2_1;

pub use qf8_4_type::qf8_4;
pub use qf8_4_1_type::qf8_4_1;
pub use qf8_4_2_type::qf8_4_2;

pub use qf8_8_type::qf8_8;
pub use qf8_8_1_type::qf8_8_1;
pub use qf8_8_2_type::qf8_8_2;

pub use qf8_16_type::qf8_16;
pub use qf8_16_1_type::qf8_16_1;
pub use qf8_16_2_type::qf8_16_2;

#[derive(Default, Clone, Copy, PartialEq, Eq, Hash)]
/// Generic 8 bit float with 1 bit sign, 2 bit exponent and 5 bit mantissa.
///
/// Generic fields `M`, `R1`, `R2`, `R3` and `R4` define invert edges of the
/// 4 available precision levels which could be defined using 2 bit exponent:
///
/// 1. `[0,    M/R1)`
/// 2. `[M/R1, M/R2)`
/// 3. `[M/R2, M/R3)`
/// 4. `[M/R3, M/R4)`
///
/// In each precision range you will get 2^5 = 32 different float numbers
/// stored in mantissa. They will be uniformally distributed within the level.
///
/// It is guaranteed that the whole values space of your float is `(-M/R4, +M/R4)`.
pub struct QFloat8<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32>(u8);

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> QFloat8<M, R1, R2, R3, R4> {
    pub const RANGE_1: f64 = M as f64 / R1 as f64;
    pub const RANGE_2: f64 = M as f64 / R2 as f64;
    pub const RANGE_3: f64 = M as f64 / R3 as f64;
    pub const RANGE_4: f64 = M as f64 / R4 as f64;

    pub const PRECISION_1: f64 = Self::RANGE_1 / 32.0;
    pub const PRECISION_2: f64 = (Self::RANGE_2 - Self::RANGE_1) / 32.0;
    pub const PRECISION_3: f64 = (Self::RANGE_3 - Self::RANGE_2) / 32.0;
    pub const PRECISION_4: f64 = (Self::RANGE_4 - Self::RANGE_3) / 32.0;

    const SIGN_MASK: u8     = 0b10000000;
    const EXPONENT_MASK: u8 = 0b01100000;
    const MANTISSA_MASK: u8 = 0b00011111;

    /// Minimal value which can be stored by this generic float.
    pub const MIN: Self = Self(0b11111111);

    /// Maximal value which can be stored by this generic float.
    pub const MAX: Self = Self(0b01111111);

    /// Machine epsilon value.
    ///
    /// This is the lowest value which can be stored by this
    /// generic float type.
    pub const EPSILON: Self = Self(0b00000001);

    #[inline]
    /// Check if the sign bit is 0 for the stored float number.
    pub const fn is_positive(&self) -> bool {
        self.0 & Self::SIGN_MASK == 0
    }

    #[inline]
    /// Check if the sign bit is 1 for the stored float number.
    pub const fn is_negative(&self) -> bool {
        self.0 & Self::SIGN_MASK == Self::SIGN_MASK
    }

    #[inline]
    /// Get exponent of the stored float number.
    ///
    /// Guaranteed to return 0, 1, 2 or 3.
    pub const fn exponent(&self) -> u8 {
        (self.0 & Self::EXPONENT_MASK) >> 5
    }

    #[inline]
    /// Get mantissa of the stored float number.
    ///
    /// Guaranteed to return number from 0 to 31 including
    pub const fn mantissa(&self) -> u8 {
        self.0 & Self::MANTISSA_MASK
    }

    #[inline]
    /// Parse given double precision machine float number
    /// into the current generic representation.
    ///
    /// This is a constant function and can be executed in
    /// compile time context.
    pub const fn from_f64(float: f64) -> Self {
        let (abs_float, sign) = if float.is_sign_positive() {
            (float, 0)
        } else {
            (-float, Self::SIGN_MASK)
        };

        /// Round float number into the integer.
        const fn round(value: f64) -> u8 {
            let int = value as u8;

            if value - (int as f64) < 0.5 {
                int
            } else {
                int + 1
            }
        }

        /// Return clamp is value is greater than it, otherwise value.
        const fn clamp(value: u8, clamp: u8) -> u8 {
            if value > clamp {
                clamp
            } else {
                value
            }
        }

        // Exponent 1: [0, M/R1)
        if abs_float < Self::RANGE_1 {
            let mantissa = round(abs_float / Self::PRECISION_1);

            Self(sign | clamp(mantissa, Self::MANTISSA_MASK))
        }

        // Exponent 2: [M/R1, M/R2)
        else if abs_float < Self::RANGE_2 {
            let mantissa = round((abs_float - Self::RANGE_1) / Self::PRECISION_2);

            Self(sign | 0b00100000 | clamp(mantissa, Self::MANTISSA_MASK))
        }

        // Exponent 3: [M/R2, M/R3)
        else if abs_float < Self::RANGE_3 {
            let mantissa = round((abs_float - Self::RANGE_2) / Self::PRECISION_3);

            Self(sign | 0b01000000 | clamp(mantissa, Self::MANTISSA_MASK))
        }

        // Exponent 4: [M/R3, M/R4)
        else if abs_float < Self::RANGE_4 {
            let mantissa = round((abs_float - Self::RANGE_3) / Self::PRECISION_4);

            Self(sign | 0b01100000 | clamp(mantissa, Self::MANTISSA_MASK))
        }

        // Values greater than the M/R4 are clamped to the max
        // value that can be stored, respecting their sign.
        else {
            Self(sign | 0b01111111)
        }
    }

    #[inline]
    /// Convert current generic float number into machine
    /// double precision float.
    ///
    /// This is a constant function and can be executed in
    /// compile time context.
    pub const fn into_f64(&self) -> f64 {
        let exponent = self.exponent();
        let mantissa = self.mantissa() as f64;

        let value = match exponent {
            0 =>                 mantissa * Self::PRECISION_1,
            1 => Self::RANGE_1 + mantissa * Self::PRECISION_2,
            2 => Self::RANGE_2 + mantissa * Self::PRECISION_3,
            3 => Self::RANGE_3 + mantissa * Self::PRECISION_4,

            _ => unreachable!()
        };

        if self.is_positive() { value } else { -value }
    }

    #[inline]
    /// Return positive value of the current float number.
    pub const fn abs(&self) -> Self {
        Self(self.0 & !Self::SIGN_MASK)
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> From<f32> for QFloat8<M, R1, R2, R3, R4> {
    #[inline]
    fn from(value: f32) -> Self {
        Self::from_f64(value as f64)
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> From<f64> for QFloat8<M, R1, R2, R3, R4> {
    #[inline]
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> std::fmt::Display for QFloat8<M, R1, R2, R3, R4> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.as_f64();

        // For integers (0.0, 1.0, etc.)
        if value.fract() == 0.0 {
            write!(f, "{value:.1}")
        }

        // For floats (0.5, -0.3, etc.)
        else {
            write!(f, "{value}")
        }
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> std::fmt::Debug for QFloat8<M, R1, R2, R3, R4> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QFloat8({self})")
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> Neg for QFloat8<M, R1, R2, R3, R4> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(self.0 ^ Self::SIGN_MASK)
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> Add for QFloat8<M, R1, R2, R3, R4> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_float(self.as_f64() + rhs.as_f64())
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> Sub for QFloat8<M, R1, R2, R3, R4> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_float(self.as_f64() - rhs.as_f64())
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> Mul for QFloat8<M, R1, R2, R3, R4> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_float(self.as_f64() * rhs.as_f64())
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> Div for QFloat8<M, R1, R2, R3, R4> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self::from_float(self.as_f64() / rhs.as_f64())
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> AddAssign for QFloat8<M, R1, R2, R3, R4> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = (*self + rhs).0;
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> SubAssign for QFloat8<M, R1, R2, R3, R4> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = (*self - rhs).0;
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> MulAssign for QFloat8<M, R1, R2, R3, R4> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = (*self * rhs).0;
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> DivAssign for QFloat8<M, R1, R2, R3, R4> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.0 = (*self / rhs).0;
    }
}

impl<const M: u32, const R1: u32, const R2: u32, const R3: u32, const R4: u32> Float for QFloat8<M, R1, R2, R3, R4> {
    const ZERO: Self = QFloat8::<M, R1, R2, R3, R4>::from_f64(0.0);
    const HALF: Self = QFloat8::<M, R1, R2, R3, R4>::from_f64(0.5);
    const ONE: Self  = QFloat8::<M, R1, R2, R3, R4>::from_f64(1.0);

    const MIN: Self     = QFloat8::<M, R1, R2, R3, R4>::MIN;
    const MAX: Self     = QFloat8::<M, R1, R2, R3, R4>::MAX;
    const EPSILON: Self = QFloat8::<M, R1, R2, R3, R4>::EPSILON;

    #[inline]
    fn as_f32(&self) -> f32 {
        Self::into_f64(self) as f32
    }

    #[inline]
    fn as_f64(&self) -> f64 {
        Self::into_f64(self)
    }

    #[inline]
    fn from_float<F: Float>(float: F) -> Self {
        Self::from_f64(float.as_f64())
    }

    #[inline]
    fn to_bytes(&self) -> Box<[u8]> {
        Box::new([self.0])
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), 1, "QFloat8 must be stored in 1 byte");

        QFloat8(bytes[0])
    }

    // =================================== Arithmetic functions ===================================

    #[inline]
    fn is_positive(&self) -> bool {
        Self::is_positive(self)
    }

    #[inline]
    fn is_negative(&self) -> bool {
        Self::is_negative(self)
    }

    #[inline]
    fn abs(&self) -> Self {
        Self::abs(self)
    }

    // Other functions are too difficult to optimize.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_values() {
        assert_eq!(qf8_2::from_float(0.0).as_f32(), 0.0);
        assert_eq!(qf8_2::from_float(0.25).as_f32(), 0.25);
        assert_eq!(qf8_2::from_float(0.5).as_f32(), 0.5);
        assert_eq!(qf8_2::from_float(1.0).as_f32(), 1.0);
        assert_eq!(qf8_2::from_float(2.0).as_f32(), qf8_2::MAX.as_f32());
        assert_eq!(qf8_2::from_float(3.0).as_f32(), qf8_2::MAX.as_f32());

        assert_eq!(qf8_2::from_float(-0.0).as_f32(), -0.0);
        assert_eq!(qf8_2::from_float(-0.25).as_f32(), -0.25);
        assert_eq!(qf8_2::from_float(-0.5).as_f32(), -0.5);
        assert_eq!(qf8_2::from_float(-1.0).as_f32(), -1.0);
        assert_eq!(qf8_2::from_float(-2.0).as_f32(), qf8_2::MIN.as_f32());
        assert_eq!(qf8_2::from_float(-3.0).as_f32(), qf8_2::MIN.as_f32());
    }

    #[test]
    fn display() {
        assert_eq!(qf8_2::ZERO.to_string(), "0.0");
        assert_eq!(qf8_2::HALF.to_string(), "0.5");
        assert_eq!(qf8_2::ONE.to_string(),  "1.0");

        assert_eq!((-qf8_2::ZERO).to_string(), "-0.0");
        assert_eq!((-qf8_2::HALF).to_string(), "-0.5");
        assert_eq!((-qf8_2::ONE).to_string(),  "-1.0");

        assert_eq!(qf8_2::from_float(0.25).to_string(), "0.25");
        assert_eq!(qf8_2::from_float(0.0625).to_string(), "0.0625");

        assert_eq!(qf8_2::from_float(-0.25).to_string(), "-0.25");
        assert_eq!(qf8_2::from_float(-0.0625).to_string(), "-0.0625");
    }

    #[test]
    fn from_float() {
        assert_eq!(qf8_2::ZERO, qf8_2::from_float(0.0));
        assert_eq!(qf8_2::HALF, qf8_2::from_float(0.5));
        assert_eq!(qf8_2::ONE,  qf8_2::from_float(1.0));

        assert_eq!(-qf8_2::ZERO, qf8_2::from_float(-0.0));
        assert_eq!(-qf8_2::HALF, qf8_2::from_float(-0.5));
        assert_eq!(-qf8_2::ONE,  qf8_2::from_float(-1.0));

        for exponent in 0b00..=0b11 {
            // print!("/// ### Exponent {exponent}:\n///\n/// ```text,ignore\n/// ");

            for mantissa in 0b00000..=0b11111 {
                let positive = QFloat8::<11, 40, 30, 20, 10>((exponent << 5) | mantissa);
                let negative = -positive;

                // The same number can be encoded differently so we should check
                // float representations here.
                assert_eq!(qf8_1::from_float(positive.as_f32()).as_f32(), positive.as_f32());
                assert_eq!(qf8_1::from_float(negative.as_f32()).as_f32(), negative.as_f32());

                // print!("{:.08}, ", positive.as_f32());
            }

            // print!("\n/// ```\n///\n");
        }

        // assert!(false);
    }

    #[test]
    fn arithmetics() {
        assert_eq!(qf8_2::MAX * qf8_2::ZERO, qf8_2::ZERO);
        assert_eq!(qf8_2::MAX * -qf8_2::ONE, qf8_2::MIN);
        assert_eq!(qf8_2::HALF + qf8_2::HALF, qf8_2::ONE);
        assert_eq!(-qf8_2::HALF + qf8_2::HALF, qf8_2::ZERO);
        assert_eq!(qf8_2::ONE - qf8_2::HALF, qf8_2::HALF);
        assert_eq!(qf8_2::HALF - qf8_2::ONE, -qf8_2::HALF);
        assert_eq!(qf8_2::MAX - qf8_2::MAX, qf8_2::ZERO);

        assert!((qf8_2::HALF * qf8_2::from_float(2.0)).as_f32() > 0.8);
        assert!((qf8_2::ONE / qf8_2::from_f64(2.0)).as_f32() < 0.8);
    }
}
