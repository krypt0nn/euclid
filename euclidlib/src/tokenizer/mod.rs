use crate::prelude::*;

pub mod parser;
pub mod generic_model;
pub mod sized_model;
pub mod database;

pub mod prelude {
    pub use super::parser::Parser as DocumentsParser;
    pub use super::generic_model::GenericModel as GenericWordEmbeddingsModel;
    pub use super::sized_model::SizedModel as WordEmbeddingsModel;

    pub use super::database::prelude::*;

    pub use super::cosine_similarity;
}

/// Calculate cosine similarity between two vectors.
///
/// Return value in `[-1.0, 1.0]` range where 1.0 means fully equal.
pub fn cosine_similarity<const N: usize, F: Float>(word_1: &[F], word_2: &[F]) -> f64 {
    let mut distance = 0.0;
    let mut len_1 = 0.0;
    let mut len_2 = 0.0;

    for i in 0..N {
        let word_1 = word_1.get(i).copied().unwrap_or(F::ZERO).as_f64();
        let word_2 = word_2.get(i).copied().unwrap_or(F::ZERO).as_f64();

        distance += word_1 * word_2;

        len_1 += word_1.powi(2);
        len_2 += word_2.powi(2);
    }

    distance / (len_1.sqrt() * len_2.sqrt())
}
