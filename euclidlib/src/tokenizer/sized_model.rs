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

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Information about the dynamically sized model.
pub struct SizedModelParams {
    pub input_tokens: usize,
    pub embedding_dimensions: usize,
    pub embedding_context_radius: usize,
    pub parameters: usize
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Information about the loss calculated over the long sequence of tokens.
pub struct SizedModelLoss<F: Float> {
    pub min_loss: F,
    pub mean_loss: F,
    pub max_loss: F,
    pub total_loss: F
}

#[derive(Debug, Clone, PartialEq)]
/// Dynamically sized word embeddings language model.
pub enum SizedModel<F: Float> {
    // ! Be accurate at changing following constants because they're used
    // ! in other parts of the code as well and you will have to find and
    // ! update them there too.

    /// 1K tokens, 32 dimensions, 4 context tokens, 66K parameters.
    Tiny(GenericWordEmbeddingsModel<1024, 8, F>),

    /// 4K tokens, 64 dimensions, 4 context tokens, 500K parameters.
    Small(GenericWordEmbeddingsModel<4096, 16, F>),

    /// 16K tokens, 128 dimensions, 8 context tokens, 4M parameters.
    Medium(GenericWordEmbeddingsModel<16384, 24, F>),

    /// 65K tokens, 256 dimensions, 8 context tokens, 33M parameters.
    Large(GenericWordEmbeddingsModel<65536, 32, F>),

    /// 250K tokens, 512 dimensions, 12 context tokens, 269M parameters.
    Huge(GenericWordEmbeddingsModel<262144, 48, F>),

    /// 1M tokens, 1024 dimensions, 12 context tokens, 2B parameters.
    Giant(GenericWordEmbeddingsModel<1048576, 64, F>)
}

impl<F: Float> SizedModel<F> {
    #[inline]
    /// Create new tiny model with random parameters.
    pub fn random_tiny() -> Self {
        Self::Tiny(GenericWordEmbeddingsModel::<1024, 8, F>::random())
    }

    #[inline]
    /// Create new small model with random parameters.
    pub fn random_small() -> Self {
        Self::Small(GenericWordEmbeddingsModel::<4096, 16, F>::random())
    }

    #[inline]
    /// Create new medium model with random parameters.
    pub fn random_medium() -> Self {
        Self::Medium(GenericWordEmbeddingsModel::<16384, 24, F>::random())
    }

    #[inline]
    /// Create new large model with random parameters.
    pub fn random_large() -> Self {
        Self::Large(GenericWordEmbeddingsModel::<65536, 32, F>::random())
    }

    #[inline]
    /// Create new huge model with random parameters.
    pub fn random_huge() -> Self {
        Self::Huge(GenericWordEmbeddingsModel::<262144, 48, F>::random())
    }

    #[inline]
    /// Create new giant model with random parameters.
    pub fn random_giant() -> Self {
        Self::Giant(GenericWordEmbeddingsModel::<1048576, 64, F>::random())
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
            Self::Tiny(model)   => Self::from_generic(model.resize::<4096,    16>()),
            Self::Small(model)  => Self::from_generic(model.resize::<16384,   24>()),
            Self::Medium(model) => Self::from_generic(model.resize::<65536,   32>()),
            Self::Large(model)  => Self::from_generic(model.resize::<262144,  48>()),
            Self::Huge(model)   => Self::from_generic(model.resize::<1048576, 64>()),

            Self::Giant(_) => None
        }
    }

    /// Get information about the underlying standard generic model.
    pub const fn params(&self) -> SizedModelParams {
        match self {
            Self::Tiny(_) => SizedModelParams {
                input_tokens: 1024,
                embedding_dimensions: 8,
                embedding_context_radius: 4,
                parameters: GenericWordEmbeddingsModel::<1024, 8, F>::PARAMS
            },

            Self::Small(_) => SizedModelParams {
                input_tokens: 4096,
                embedding_dimensions: 16,
                embedding_context_radius: 4,
                parameters: GenericWordEmbeddingsModel::<4096, 16, F>::PARAMS
            },

            Self::Medium(_) => SizedModelParams {
                input_tokens: 16384,
                embedding_dimensions: 24,
                embedding_context_radius: 8,
                parameters: GenericWordEmbeddingsModel::<16384, 24, F>::PARAMS
            },

            Self::Large(_) => SizedModelParams {
                input_tokens: 65536,
                embedding_dimensions: 32,
                embedding_context_radius: 8,
                parameters: GenericWordEmbeddingsModel::<65536, 32, F>::PARAMS
            },

            Self::Huge(_) => SizedModelParams {
                input_tokens: 262144,
                embedding_dimensions: 48,
                embedding_context_radius: 12,
                parameters: GenericWordEmbeddingsModel::<262144, 48, F>::PARAMS
            },

            Self::Giant(_) => SizedModelParams {
                input_tokens: 1048576,
                embedding_dimensions: 64,
                embedding_context_radius: 12,
                parameters: GenericWordEmbeddingsModel::<1048576, 64, F>::PARAMS
            }
        }
    }

    /// Encode given token to the embedding vector.
    pub fn encode(&self, token: usize, device: &impl Device) -> Vec<F> {
        match self {
            Self::Tiny(model)   => model.encode(token, device).to_vec(),
            Self::Small(model)  => model.encode(token, device).to_vec(),
            Self::Medium(model) => model.encode(token, device).to_vec(),
            Self::Large(model)  => model.encode(token, device).to_vec(),
            Self::Huge(model)   => model.encode(token, device).to_vec(),
            Self::Giant(model)  => model.encode(token, device).to_vec()
        }
    }

    /// Decode given embedding vector to the token.
    pub fn decode(&self, embedding: &[F], device: &impl Device) -> usize {
        match self {
            Self::Tiny(model)   => model.decode(scale_slice(embedding), device),
            Self::Small(model)  => model.decode(scale_slice(embedding), device),
            Self::Medium(model) => model.decode(scale_slice(embedding), device),
            Self::Large(model)  => model.decode(scale_slice(embedding), device),
            Self::Huge(model)   => model.decode(scale_slice(embedding), device),
            Self::Giant(model)  => model.decode(scale_slice(embedding), device)
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
        policy: &mut BackpropagationSnapshot<PARAMS, F>,
        device: &mut impl Device
    )
    where
        // This incredible where statement is needed to fix const generics which are BROKEN!!!!!!!!

        [(); { EncoderDecoder::<1024, 8, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<1024, 8, F>::PARAMS }]: Sized,
        [(); { Layer::<8, 1024, F>::PARAMS }]: Sized,
        [(); { Neuron::<1024, F>::PARAMS }]: Sized,
        [(); { Neuron::<8, F>::PARAMS }]: Sized,

        [(); { EncoderDecoder::<4096, 16, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<4096, 16, F>::PARAMS }]: Sized,
        [(); { Layer::<16, 4096, F>::PARAMS }]: Sized,
        [(); { Neuron::<4096, F>::PARAMS }]: Sized,
        [(); { Neuron::<16, F>::PARAMS }]: Sized,

        [(); { EncoderDecoder::<16384, 24, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<16384, 24, F>::PARAMS }]: Sized,
        [(); { Layer::<24, 16384, F>::PARAMS }]: Sized,
        [(); { Neuron::<16384, F>::PARAMS }]: Sized,
        [(); { Neuron::<24, F>::PARAMS }]: Sized,

        [(); { EncoderDecoder::<65536, 32, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<65536, 32, F>::PARAMS }]: Sized,
        [(); { Layer::<32, 65536, F>::PARAMS }]: Sized,
        [(); { Neuron::<65536, F>::PARAMS }]: Sized,
        [(); { Neuron::<32, F>::PARAMS }]: Sized,

        [(); { EncoderDecoder::<262144, 48, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<262144, 48, F>::PARAMS }]: Sized,
        [(); { Layer::<48, 262144, F>::PARAMS }]: Sized,
        [(); { Neuron::<262144, F>::PARAMS }]: Sized,
        [(); { Neuron::<48, F>::PARAMS }]: Sized,

        [(); { EncoderDecoder::<1048576, 64, F>::MODEL_PARAMS }]: Sized,
        [(); { Layer::<1048576, 64, F>::PARAMS }]: Sized,
        [(); { Layer::<64, 1048576, F>::PARAMS }]: Sized,
        [(); { Neuron::<1048576, F>::PARAMS }]: Sized,
        [(); { Neuron::<64, F>::PARAMS }]: Sized
    {
        // policy.window() will zero-allocate new params gradients if BackpropagationSnapshot is too small.
        match self {
            Self::Tiny(model)   => policy.window::<{ EncoderDecoder::<1024,    8,  F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<4>(tokens, &mut policy, device)),
            Self::Small(model)  => policy.window::<{ EncoderDecoder::<4096,    16, F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<4>(tokens, &mut policy, device)),
            Self::Medium(model) => policy.window::<{ EncoderDecoder::<16384,   24, F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<8>(tokens, &mut policy, device)),
            Self::Large(model)  => policy.window::<{ EncoderDecoder::<65536,   32, F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<8>(tokens, &mut policy, device)),
            Self::Huge(model)   => policy.window::<{ EncoderDecoder::<262144,  48, F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<12>(tokens, &mut policy, device)),
            Self::Giant(model)  => policy.window::<{ EncoderDecoder::<1048576, 64, F>::MODEL_PARAMS }, _>(0, move |mut policy| model.train::<12>(tokens, &mut policy, device))
        }
    }

    /// Calculate loss between expected token and an actual prediction of the model.
    pub fn loss(&self, input_tokens: &[usize], expected_output: usize, device: &impl Device) -> F {
        match self {
            Self::Tiny(model)   => model.loss(input_tokens, expected_output, device),
            Self::Small(model)  => model.loss(input_tokens, expected_output, device),
            Self::Medium(model) => model.loss(input_tokens, expected_output, device),
            Self::Large(model)  => model.loss(input_tokens, expected_output, device),
            Self::Huge(model)   => model.loss(input_tokens, expected_output, device),
            Self::Giant(model)  => model.loss(input_tokens, expected_output, device)
        }
    }

    /// Calculate loss statistics on the given tokens sequence.
    pub fn total_loss(&self, tokens: &[usize], device: &impl Device) -> SizedModelLoss<F> {
        fn window_over<F: Float>(tokens: &[usize], radius: usize, model: &SizedModel<F>, device: &impl Device) -> SizedModelLoss<F> {
            let mut window = vec![0; radius * 2];

            let n = tokens.len();

            let mut loss_stats = SizedModelLoss::default();
            let mut k = 1;

            for i in radius..n - radius {
                window[..radius].copy_from_slice(&tokens[i - radius..i]);
                window[radius..].copy_from_slice(&tokens[i + 1..i + 1 + radius]);

                let loss = model.loss(&window, tokens[i], device);

                if k == 1 {
                    loss_stats.min_loss = loss;
                }

                if loss < loss_stats.min_loss {
                    loss_stats.min_loss = loss;
                } else if loss > loss_stats.max_loss {
                    loss_stats.max_loss = loss;
                }

                loss_stats.total_loss += loss;
                loss_stats.mean_loss = loss_stats.total_loss / F::from_float(k as f32);

                k += 1;
            }

            loss_stats
        }

        let params = self.params();

        window_over(tokens, params.embedding_context_radius, self, device)
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<1024, 8, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<1024, 8, F>) -> Self {
        Self::Tiny(model)
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<4096, 16, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<4096, 16, F>) -> Self {
        Self::Small(model)
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<16384, 24, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<16384, 24, F>) -> Self {
        Self::Medium(model)
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<65536, 32, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<65536, 32, F>) -> Self {
        Self::Large(model)
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<262144, 48, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<262144, 48, F>) -> Self {
        Self::Huge(model)
    }
}

impl<F: Float> From<GenericWordEmbeddingsModel<1048576, 64, F>> for SizedModel<F> {
    #[inline]
    fn from(model: GenericWordEmbeddingsModel<1048576, 64, F>) -> Self {
        Self::Giant(model)
    }
}
