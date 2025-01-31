use crate::prelude::*;

#[derive(Debug, Clone, PartialEq)]
/// Dynamically sized word embeddings language model.
pub enum SizedModel<F: Float> {
    /// 1K tokens, 16 dimensions, 34K parameters.
    Tiny(GenericWordEmbeddingsModel<1024, 16, F>),

    /// 4K tokens, 64 dimensions, 500K parameters.
    Small(GenericWordEmbeddingsModel<4096, 64, F>),

    /// 16K tokens, 128 dimensions, 4M parameters.
    Medium(GenericWordEmbeddingsModel<16384, 128, F>),

    /// 65K tokens, 256 dimensions, 33M parameters.
    Large(GenericWordEmbeddingsModel<65536, 256, F>),

    /// 250K tokens, 512 dimensions, 269M parameters.
    Huge(GenericWordEmbeddingsModel<262144, 512, F>),

    /// 1M tokens, 1024 dimensions, 2B parameters.
    Giant(GenericWordEmbeddingsModel<1048576, 1024, F>)
}

impl<F: Float> SizedModel<F> {
    /// Convert given generic word embeddings model into dynamically sized.
    ///
    /// This function will resize provided generic model to the closest
    /// dynamically sized type, trying to keep original precision.
    ///
    /// Return `None` if `TOKENS_NUM` is too large.
    pub fn from_generic<
        const TOKENS_NUM: usize,
        const EMBEDDING_SIZE: usize
    >(model: GenericWordEmbeddingsModel<TOKENS_NUM, EMBEDDING_SIZE, F>) -> Option<Self> {
        if TOKENS_NUM <= 1024 {
            Some(Self::Tiny(model.resize()))
        }

        else if TOKENS_NUM <= 4096 {
            Some(Self::Small(model.resize()))
        }

        else if TOKENS_NUM <= 16384 {
            Some(Self::Medium(model.resize()))
        }

        else if TOKENS_NUM <= 65536 {
            Some(Self::Large(model.resize()))
        }

        else if TOKENS_NUM <= 262144 {
            Some(Self::Huge(model.resize()))
        }

        else if TOKENS_NUM <= 1048576 {
            Some(Self::Giant(model.resize()))
        }

        else {
            None
        }
    }

    /// Get embedding of a given token.
    pub fn get_embedding(&self, token: usize) -> Vec<F> {
        match self {
            Self::Tiny(model)   => model.get_embedding(token).to_vec(),
            Self::Small(model)  => model.get_embedding(token).to_vec(),
            Self::Medium(model) => model.get_embedding(token).to_vec(),
            Self::Large(model)  => model.get_embedding(token).to_vec(),
            Self::Huge(model)   => model.get_embedding(token).to_vec(),
            Self::Giant(model)  => model.get_embedding(token).to_vec()
        }
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<1024, 16, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<1024, 16, F>) -> Self {
        Self::Tiny(model)
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<4096, 64, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<4096, 64, F>) -> Self {
        Self::Small(model)
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<16384, 128, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<16384, 128, F>) -> Self {
        Self::Medium(model)
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<65536, 256, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<65536, 256, F>) -> Self {
        Self::Large(model)
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<262144, 512, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<262144, 512, F>) -> Self {
        Self::Huge(model)
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<1048576, 1024, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<1048576, 1024, F>) -> Self {
        Self::Giant(model)
    }
}
