use crate::prelude::*;

fn scale_slice<const LEN: usize, F: Float>(slice: &[F]) -> Box<[F; LEN]> {
    let mut scaled = unsafe {
        alloc_fixed_heap_array_with(F::ZERO)
            .expect("Failed to allocate memory for scaled slice")
    };

    let slice_len = slice.len();

    if LEN >= slice_len {
        scaled[..slice_len].copy_from_slice(slice);
    } else {
        scaled.copy_from_slice(&slice[..LEN]);
    }

    scaled
}

#[derive(Debug, Clone, PartialEq)]
/// Dynamically sized word embeddings language model.
pub enum SizedModel<F: Float> {
    // ! Be accurate at changing following constants because they're used
    // ! in other parts of the code as well and you will have to find and
    // ! update them there too.

    /// 1K tokens, 32 dimensions, 4 context tokens, 66K parameters.
    Tiny(GenericWordEmbeddingsModel<1024, 32, F>),

    /// 4K tokens, 64 dimensions, 4 context tokens, 500K parameters.
    Small(GenericWordEmbeddingsModel<4096, 64, F>),

    /// 16K tokens, 128 dimensions, 8 context tokens, 4M parameters.
    Medium(GenericWordEmbeddingsModel<16384, 128, F>),

    /// 65K tokens, 256 dimensions, 8 context tokens, 33M parameters.
    Large(GenericWordEmbeddingsModel<65536, 256, F>),

    /// 250K tokens, 512 dimensions, 12 context tokens, 269M parameters.
    Huge(GenericWordEmbeddingsModel<262144, 512, F>),

    /// 1M tokens, 1024 dimensions, 12 context tokens, 2B parameters.
    Giant(GenericWordEmbeddingsModel<1048576, 1024, F>)
}

impl<F: Float> SizedModel<F> {
    #[inline]
    /// Create new tiny model with random parameters.
    pub fn random_tiny() -> Self {
        Self::Tiny(GenericWordEmbeddingsModel::<1024, 32, F>::random())
    }

    #[inline]
    /// Create new small model with random parameters.
    pub fn random_small() -> Self {
        Self::Small(GenericWordEmbeddingsModel::<4096, 64, F>::random())
    }

    #[inline]
    /// Create new medium model with random parameters.
    pub fn random_medium() -> Self {
        Self::Medium(GenericWordEmbeddingsModel::<16384, 128, F>::random())
    }

    #[inline]
    /// Create new large model with random parameters.
    pub fn random_large() -> Self {
        Self::Large(GenericWordEmbeddingsModel::<65536, 256, F>::random())
    }

    #[inline]
    /// Create new huge model with random parameters.
    pub fn random_huge() -> Self {
        Self::Huge(GenericWordEmbeddingsModel::<262144, 512, F>::random())
    }

    #[inline]
    /// Create new giant model with random parameters.
    pub fn random_giant() -> Self {
        Self::Giant(GenericWordEmbeddingsModel::<1048576, 1024, F>::random())
    }

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
    pub fn resize<
        const NEW_TOKENS_NUM: usize,
        const NEW_EMBEDDING_SIZE: usize
    >(&self) -> Option<Self> {
        match self {
            Self::Tiny(model)   => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>()),
            Self::Small(model)  => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>()),
            Self::Medium(model) => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>()),
            Self::Large(model)  => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>()),
            Self::Huge(model)   => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>()),
            Self::Giant(model)  => Self::from_generic(model.resize::<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE>())
        }
    }

    /// Upscale the model to the higher resolution.
    pub fn upscale(&self) -> Option<Self> {
        match self {
            Self::Tiny(model)   => Self::from_generic(model.resize::<4096,    64>()),
            Self::Small(model)  => Self::from_generic(model.resize::<16384,   128>()),
            Self::Medium(model) => Self::from_generic(model.resize::<65536,   256>()),
            Self::Large(model)  => Self::from_generic(model.resize::<262144,  512>()),
            Self::Huge(model)   => Self::from_generic(model.resize::<1048576, 1024>()),

            Self::Giant(_) => None
        }
    }

    /// Get amount of input tokens, embedding dimensions and parameters
    /// in the current model size variant.
    pub fn params(&self) -> (usize, usize, usize) {
        match self {
            Self::Tiny(_)   => (1024,    32,   GenericWordEmbeddingsModel::<1024,    32,   F>::PARAMS),
            Self::Small(_)  => (4096,    64,   GenericWordEmbeddingsModel::<4096,    64,   F>::PARAMS),
            Self::Medium(_) => (16384,   128,  GenericWordEmbeddingsModel::<16384,   128,  F>::PARAMS),
            Self::Large(_)  => (65536,   256,  GenericWordEmbeddingsModel::<65536,   256,  F>::PARAMS),
            Self::Huge(_)   => (262144,  512,  GenericWordEmbeddingsModel::<262144,  512,  F>::PARAMS),
            Self::Giant(_)  => (1048576, 1024, GenericWordEmbeddingsModel::<1048576, 1024, F>::PARAMS)
        }
    }

    /// Encode given token to the embedding vector.
    pub fn encode(&self, token: usize) -> Vec<F> {
        match self {
            Self::Tiny(model)   => model.encode(token).to_vec(),
            Self::Small(model)  => model.encode(token).to_vec(),
            Self::Medium(model) => model.encode(token).to_vec(),
            Self::Large(model)  => model.encode(token).to_vec(),
            Self::Huge(model)   => model.encode(token).to_vec(),
            Self::Giant(model)  => model.encode(token).to_vec()
        }
    }

    /// Decode given embedding vector to the token.
    pub fn decode(&self, embedding: &[F]) -> usize {
        match self {
            Self::Tiny(model)   => model.decode(scale_slice(embedding)),
            Self::Small(model)  => model.decode(scale_slice(embedding)),
            Self::Medium(model) => model.decode(scale_slice(embedding)),
            Self::Large(model)  => model.decode(scale_slice(embedding)),
            Self::Huge(model)   => model.decode(scale_slice(embedding)),
            Self::Giant(model)  => model.decode(scale_slice(embedding))
        }
    }

    #[allow(unused_braces)]
    /// Train the model once on given tokens slice with given radius and policy.
    ///
    /// Since `Backpropagation` struct must be statically allocated at compile time
    /// you have to make sure to allocate enough space for it to use AdamW optimization
    /// properly. Otherwise moving gradients would be set to defaults which would
    /// decrease training quality.
    pub fn train<const PARAMS: usize>(
        &mut self,
        tokens: &[usize],
        policy: &mut BackpropagationSnapshot<'_, PARAMS, F>
    )
    where
        // This incredible where statement is needed to fix const generics which are BROKEN!!!!!!!!

        [(); { EncoderDecoder::<1024, 32, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<1024, 32, F>::PARAMS }]: Sized,
        [(); { Layer::<32, 1024, F>::PARAMS }]: Sized,
        [(); { Neuron::<1024, F>::PARAMS }]: Sized,
        [(); { Neuron::<32, F>::PARAMS }]: Sized,

        [(); { EncoderDecoder::<4096, 64, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<4096, 64, F>::PARAMS }]: Sized,
        [(); { Layer::<64, 4096, F>::PARAMS }]: Sized,
        [(); { Neuron::<4096, F>::PARAMS }]: Sized,
        [(); { Neuron::<64, F>::PARAMS }]: Sized,

        [(); { EncoderDecoder::<16384, 128, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<16384, 128, F>::PARAMS }]: Sized,
        [(); { Layer::<128, 16384, F>::PARAMS }]: Sized,
        [(); { Neuron::<16384, F>::PARAMS }]: Sized,
        [(); { Neuron::<128, F>::PARAMS }]: Sized,

        [(); { EncoderDecoder::<65536, 256, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<65536, 256, F>::PARAMS }]: Sized,
        [(); { Layer::<256, 65536, F>::PARAMS }]: Sized,
        [(); { Neuron::<65536, F>::PARAMS }]: Sized,
        [(); { Neuron::<256, F>::PARAMS }]: Sized,

        [(); { EncoderDecoder::<262144, 512, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<262144, 512, F>::PARAMS }]: Sized,
        [(); { Layer::<512, 262144, F>::PARAMS }]: Sized,
        [(); { Neuron::<262144, F>::PARAMS }]: Sized,
        [(); { Neuron::<512, F>::PARAMS }]: Sized,

        [(); { EncoderDecoder::<1048576, 1024, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<1048576, 1024, F>::PARAMS }]: Sized,
        [(); { Layer::<1024, 1048576, F>::PARAMS }]: Sized,
        [(); { Neuron::<1048576, F>::PARAMS }]: Sized,
        [(); { Neuron::<1024, F>::PARAMS }]: Sized
    {
        // policy.window() will zero-allocate new params gradients if BackpropagationSnapshot is too small.
        match self {
            Self::Tiny(model)   => policy.window::<{ EncoderDecoder::<1024,    32,   F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<4>(tokens, &mut policy)),
            Self::Small(model)  => policy.window::<{ EncoderDecoder::<4096,    64,   F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<4>(tokens, &mut policy)),
            Self::Medium(model) => policy.window::<{ EncoderDecoder::<16384,   128,  F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<8>(tokens, &mut policy)),
            Self::Large(model)  => policy.window::<{ EncoderDecoder::<65536,   256,  F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<8>(tokens, &mut policy)),
            Self::Huge(model)   => policy.window::<{ EncoderDecoder::<262144,  512,  F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<12>(tokens, &mut policy)),
            Self::Giant(model)  => policy.window::<{ EncoderDecoder::<1048576, 1024, F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<12>(tokens, &mut policy))
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

impl<F: Float> From<GenericWordEmbeddingsModel<1024, 32, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<1024, 32, F>) -> Self {
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
