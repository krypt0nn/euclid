use crate::prelude::*;

#[derive(Debug, Clone, PartialEq)]
/// Dynamically sized word embeddings language model.
pub enum SizedModel<F: Float> {
    // ! Be accurate at changing following constants because they're used
    // ! in other parts of the code as well and you will have to find and
    // ! update them there too.

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

    /// Resize model by truncating neurons and weights of the neurons layers or repeating them.
    pub fn resize<const NEW_TOKENS_NUM: usize, const NEW_EMBEDDING_SIZE: usize>(&self) -> Option<Self> {
        match self {
            Self::Tiny(model)   => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>()),
            Self::Small(model)  => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>()),
            Self::Medium(model) => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>()),
            Self::Large(model)  => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>()),
            Self::Huge(model)   => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>()),
            Self::Giant(model)  => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>())
        }
    }

    /// Get amount of parameters in the current model size variant.
    pub fn params(&self) -> usize {
        match self {
            Self::Tiny(_)   => GenericWordEmbeddingsModel::<1024,    16,   F>::PARAMS,
            Self::Small(_)  => GenericWordEmbeddingsModel::<4096,    64,   F>::PARAMS,
            Self::Medium(_) => GenericWordEmbeddingsModel::<16384,   128,  F>::PARAMS,
            Self::Large(_)  => GenericWordEmbeddingsModel::<65536,   256,  F>::PARAMS,
            Self::Huge(_)   => GenericWordEmbeddingsModel::<262144,  512,  F>::PARAMS,
            Self::Giant(_)  => GenericWordEmbeddingsModel::<1048576, 1024, F>::PARAMS
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

    /// Find output token with the greatest probability of occurence
    /// among the given group of input tokens.
    pub fn predict(&self, input_tokens: &[usize]) -> usize {
        match self {
            Self::Tiny(model)   => model.predict(input_tokens),
            Self::Small(model)  => model.predict(input_tokens),
            Self::Medium(model) => model.predict(input_tokens),
            Self::Large(model)  => model.predict(input_tokens),
            Self::Huge(model)   => model.predict(input_tokens),
            Self::Giant(model)  => model.predict(input_tokens)
        }
    }

    /// Calculate cosine distance between two word embeddings.
    ///
    /// Closer two words to each other - lower the output value.
    pub fn distance(&self, word_1: &[F], word_2: &[F]) -> f64 {
        fn scale_slice<const LEN: usize, F: Float>(slice: &[F]) -> [F; LEN] {
            let mut scaled = [F::ZERO; LEN];

            let slice_len = slice.len();

            if LEN >= slice_len {
                scaled[..slice_len].copy_from_slice(slice);
            } else {
                scaled.copy_from_slice(&slice[..LEN]);
            }

            scaled
        }

        fn distance<const TOKENS_NUM: usize, const EMBEDDING_SIZE: usize, F: Float>(
            word_1: &[F],
            word_2: &[F],
            model: &GenericWordEmbeddingsModel<TOKENS_NUM, EMBEDDING_SIZE, F>
        ) -> f64 {
            model.distance(
                &scale_slice(word_1),
                &scale_slice(word_2)
            )
        }

        match self {
            Self::Tiny(model)   => distance(word_1, word_2, model),
            Self::Small(model)  => distance(word_1, word_2, model),
            Self::Medium(model) => distance(word_1, word_2, model),
            Self::Large(model)  => distance(word_1, word_2, model),
            Self::Huge(model)   => distance(word_1, word_2, model),
            Self::Giant(model)  => distance(word_1, word_2, model)
        }
    }

    /// Train the model once on given tokens slice with given radius and policy.
    /// Radius specifies amount of tokens to left and right from the target one.
    /// For example, in tokens `[0, 1, 2, 3, 4]` and `radius = 2` for target token
    /// `2` the input slice will be `[0, 1, 3, 4]`. In other words, you train
    /// the model to predict token sitting in the middle of `radius * 2` tokens.
    ///
    /// Since `Backpropagation` struct must be statically allocated at compile time
    /// you have to make sure to allocate enough space for it to use AdamW optimization
    /// properly. Otherwise moving gradients would be set to defaults which would
    /// decrease training quality.
    pub fn train<const RADIUS: usize, const PARAMS: usize>(
        &mut self,
        tokens: &[usize],
        policy: &mut BackpropagationSnapshot<'_, PARAMS, F>
    ) {
        // policy.window() will zero-allocate new params gradients if BackpropagationSnapshot is too small.
        match self {
            Self::Tiny(model)   => policy.window(0, move |mut policy| model.train::<RADIUS>(tokens, &mut policy)),
            Self::Small(model)  => policy.window(0, move |mut policy| model.train::<RADIUS>(tokens, &mut policy)),
            Self::Medium(model) => policy.window(0, move |mut policy| model.train::<RADIUS>(tokens, &mut policy)),
            Self::Large(model)  => policy.window(0, move |mut policy| model.train::<RADIUS>(tokens, &mut policy)),
            Self::Huge(model)   => policy.window(0, move |mut policy| model.train::<RADIUS>(tokens, &mut policy)),
            Self::Giant(model)  => policy.window(0, move |mut policy| model.train::<RADIUS>(tokens, &mut policy))
        }
    }

    /// Calculate loss between expected token and an actual prediction of the model.
    pub fn loss(&self, input_tokens: &[usize], expected_output: usize) -> F {
        match self {
            Self::Tiny(model)   => model.loss(input_tokens, expected_output),
            Self::Small(model)  => model.loss(input_tokens, expected_output),
            Self::Medium(model) => model.loss(input_tokens, expected_output),
            Self::Large(model)  => model.loss(input_tokens, expected_output),
            Self::Huge(model)   => model.loss(input_tokens, expected_output),
            Self::Giant(model)  => model.loss(input_tokens, expected_output)
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
