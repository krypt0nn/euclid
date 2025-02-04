use super::QFloat8;

#[allow(non_camel_case_types)]
/// Float number with `(-8.0, 8.0)` range with increased precision around `0.0`.
///
/// - 64 values in `(-0.5, 0.5)`.
/// - 64 values in `(-1.5, -0.5)` and `(0.5, 1.5)`.
/// - 64 values in `(-4.0, -1.5)` and `(1.5, 4.0)`.
/// - 64 values in `(-8.0, -4.0)` and `(4.0, 8.0)`.
///
/// Unlike `qf8_8_1` this type has increased precision for `(-0.5, 0.5)`
/// while `qf8_8_1` has uniformal distribution of 128 values
/// for whole `(-2.0, 2.0)` range.
///
/// More details are available in `QFloat8` documentation.
///
/// ## Distribution chart:
///
/// ```text,ignore
///                          ^
///                       ---|---                           64 values
///                 ------   |   ------                     64 values
///           ------         |         ------               64 values
///     ------               |               ------         64 values
/// ---+-----+-----+-----+---+---+-----+-----+-----+--->
///  -8.0  -4.0  -1.5  -0.5  0  0.5   1.5   4.0   8.0
/// ```
///
/// ## Available values:
///
/// List of values which can be stored in this float type, excluding
/// negative ones.
///
/// ### Exponent 0:
///
/// ```text,ignore
/// 0.00000000, 0.01562500, 0.03125000, 0.04687500,
/// 0.06250000, 0.07812500, 0.09375000, 0.10937500,
/// 0.12500000, 0.14062500, 0.15625000, 0.17187500,
/// 0.18750000, 0.20312500, 0.21875000, 0.23437500,
/// 0.25000000, 0.26562500, 0.28125000, 0.29687500,
/// 0.31250000, 0.32812500, 0.34375000, 0.35937500,
/// 0.37500000, 0.39062500, 0.40625000, 0.42187500,
/// 0.43750000, 0.45312500, 0.46875000, 0.48437500
/// ```
///
/// ### Exponent 1:
///
/// ```text,ignore
/// 0.50000000, 0.53125000, 0.56250000, 0.59375000,
/// 0.62500000, 0.65625000, 0.68750000, 0.71875000,
/// 0.75000000, 0.78125000, 0.81250000, 0.84375000,
/// 0.87500000, 0.90625000, 0.93750000, 0.96875000,
/// 1.00000000, 1.03125000, 1.06250000, 1.09375000,
/// 1.12500000, 1.15625000, 1.18750000, 1.21875000,
/// 1.25000000, 1.28125000, 1.31250000, 1.34375000,
/// 1.37500000, 1.40625000, 1.43750000, 1.46875000
/// ```
///
/// ### Exponent 2:
///
/// ```text,ignore
/// 1.50000000, 1.57812500, 1.65625000, 1.73437500,
/// 1.81250000, 1.89062500, 1.96875000, 2.04687500,
/// 2.12500000, 2.20312500, 2.28125000, 2.35937500,
/// 2.43750000, 2.51562500, 2.59375000, 2.67187500,
/// 2.75000000, 2.82812500, 2.90625000, 2.98437500,
/// 3.06250000, 3.14062500, 3.21875000, 3.29687500,
/// 3.37500000, 3.45312500, 3.53125000, 3.60937500,
/// 3.68750000, 3.76562500, 3.84375000, 3.92187500
/// ```
///
/// ### Exponent 3:
///
/// ```text,ignore
/// 4.00000000, 4.12500000, 4.25000000, 4.37500000,
/// 4.50000000, 4.62500000, 4.75000000, 4.87500000,
/// 5.00000000, 5.12500000, 5.25000000, 5.37500000,
/// 5.50000000, 5.62500000, 5.75000000, 5.87500000,
/// 6.00000000, 6.12500000, 6.25000000, 6.37500000,
/// 6.50000000, 6.62500000, 6.75000000, 6.87500000,
/// 7.00000000, 7.12500000, 7.25000000, 7.37500000,
/// 7.50000000, 7.62500000, 7.75000000, 7.87500000
/// ```
pub type qf8_8_2 = QFloat8<48, 96, 32, 12, 6>;
