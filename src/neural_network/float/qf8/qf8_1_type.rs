use super::QFloat8;

#[allow(non_camel_case_types)]
/// Float number with `(-1.0, 1.0)` range and uniformally distributed
/// 256 values.
///
/// More details are available in `QFloat8` documentation.
///
/// ! Note that this type doesn't have precise 1.0 representation which
/// will directly impact neurons training abilities.
///
/// ## Available values:
///
/// List of values which can be stored in this float type, excluding
/// negative ones.
///
/// ### Exponent 0:
///
/// ```text,ignore
/// 0.00000000, 0.00781250, 0.01562500, 0.02343750,
/// 0.03125000, 0.03906250, 0.04687500, 0.05468750,
/// 0.06250000, 0.07031250, 0.07812500, 0.08593750,
/// 0.09375000, 0.10156250, 0.10937500, 0.11718750,
/// 0.12500000, 0.13281250, 0.14062500, 0.14843750,
/// 0.15625000, 0.16406250, 0.17187500, 0.17968750,
/// 0.18750000, 0.19531250, 0.20312500, 0.21093750,
/// 0.21875000, 0.22656250, 0.23437500, 0.24218750
/// ```
///
/// ### Exponent 1:
///
/// ```text,ignore
/// 0.25000000, 0.25781250, 0.26562500, 0.27343750,
/// 0.28125000, 0.28906250, 0.29687500, 0.30468750,
/// 0.31250000, 0.32031250, 0.32812500, 0.33593750,
/// 0.34375000, 0.35156250, 0.35937500, 0.36718750,
/// 0.37500000, 0.38281250, 0.39062500, 0.39843750,
/// 0.40625000, 0.41406250, 0.42187500, 0.42968750,
/// 0.43750000, 0.44531250, 0.45312500, 0.46093750,
/// 0.46875000, 0.47656250, 0.48437500, 0.49218750
/// ```
///
/// ### Exponent 2:
///
/// ```text,ignore
/// 0.50000000, 0.50781250, 0.51562500, 0.52343750,
/// 0.53125000, 0.53906250, 0.54687500, 0.55468750,
/// 0.56250000, 0.57031250, 0.57812500, 0.58593750,
/// 0.59375000, 0.60156250, 0.60937500, 0.61718750,
/// 0.62500000, 0.63281250, 0.64062500, 0.64843750,
/// 0.65625000, 0.66406250, 0.67187500, 0.67968750,
/// 0.68750000, 0.69531250, 0.70312500, 0.71093750,
/// 0.71875000, 0.72656250, 0.73437500, 0.74218750
/// ```
///
/// ### Exponent 3:
///
/// ```text,ignore
/// 0.75000000, 0.75781250, 0.76562500, 0.77343750,
/// 0.78125000, 0.78906250, 0.79687500, 0.80468750,
/// 0.81250000, 0.82031250, 0.82812500, 0.83593750,
/// 0.84375000, 0.85156250, 0.85937500, 0.86718750,
/// 0.87500000, 0.88281250, 0.89062500, 0.89843750,
/// 0.90625000, 0.91406250, 0.92187500, 0.92968750,
/// 0.93750000, 0.94531250, 0.95312500, 0.96093750,
/// 0.96875000, 0.97656250, 0.98437500, 0.99218750
/// ```
pub type qf8_1 = QFloat8<3, 12, 6, 4, 3>;
