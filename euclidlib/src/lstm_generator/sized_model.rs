use crate::prelude::*;

pub const TINY_CONTEXT_WINDOW: usize   = 8;
pub const SMALL_CONTEXT_WINDOW: usize  = 16;
pub const MEDIUM_CONTEXT_WINDOW: usize = 24;
pub const LARGE_CONTEXT_WINDOW: usize  = 32;
pub const HUGE_CONTEXT_WINDOW: usize   = 48;
pub const GIANT_CONTEXT_WINDOW: usize  = 64;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Information about the dynamically sized model.
pub struct SizedModelParams {
    pub embedding_dimensions: usize,
    pub context_window: usize,
    pub parameters: usize
}

/// Dynamically sized LSTM cells recurrent generator model.
pub enum SizedModel<F: Float> {
    /// 1K tokens, 8 dimensions, 64 tokens context window, 1.6K parameters.
    Tiny(GenericLSTMGenerator<{ TINY_EMBEDDING_SIZE * TINY_CONTEXT_WINDOW }, TINY_EMBEDDING_SIZE, F>),

    /// 4K tokens, 16 dimensions, 16 tokens context window, 8.7K parameters.
    Small(GenericLSTMGenerator<{ SMALL_EMBEDDING_SIZE * SMALL_CONTEXT_WINDOW }, SMALL_EMBEDDING_SIZE, F>),

    /// 16K tokens, 24 dimensions, 24 tokens context window, 24.2K parameters.
    Medium(GenericLSTMGenerator<{ MEDIUM_EMBEDDING_SIZE * MEDIUM_CONTEXT_WINDOW }, MEDIUM_EMBEDDING_SIZE, F>),

    /// 65K tokens, 32 dimensions, 32 tokens context window, 51K parameters.
    Large(GenericLSTMGenerator<{ LARGE_EMBEDDING_SIZE * LARGE_CONTEXT_WINDOW }, LARGE_EMBEDDING_SIZE, F>),

    /// 250K tokens, 48 dimensions, 48 tokens context window, 152K parameters.
    Huge(GenericLSTMGenerator<{ HUGE_EMBEDDING_SIZE * HUGE_CONTEXT_WINDOW }, HUGE_EMBEDDING_SIZE, F>),

    /// 1M tokens, 64 dimensions, 64 tokens context window, 336K parameters.
    Giant(GenericLSTMGenerator<{ GIANT_EMBEDDING_SIZE * GIANT_CONTEXT_WINDOW }, GIANT_EMBEDDING_SIZE, F>)
}

impl<F: Float> SizedModel<F> {
    #[inline]
    /// Create new tiny model with random parameters.
    pub fn tiny() -> Self {
        Self::Tiny(GenericLSTMGenerator::<{ TINY_EMBEDDING_SIZE * TINY_CONTEXT_WINDOW }, TINY_EMBEDDING_SIZE, F>::random())
    }

    #[inline]
    /// Create new small model with random parameters.
    pub fn small() -> Self {
        Self::Small(GenericLSTMGenerator::<{ SMALL_EMBEDDING_SIZE * SMALL_CONTEXT_WINDOW }, SMALL_EMBEDDING_SIZE, F>::random())
    }

    #[inline]
    /// Create new medium model with random parameters.
    pub fn medium() -> Self {
        Self::Medium(GenericLSTMGenerator::<{ MEDIUM_EMBEDDING_SIZE * MEDIUM_CONTEXT_WINDOW }, MEDIUM_EMBEDDING_SIZE, F>::random())
    }

    #[inline]
    /// Create new large model with random parameters.
    pub fn large() -> Self {
        Self::Large(GenericLSTMGenerator::<{ LARGE_EMBEDDING_SIZE * LARGE_CONTEXT_WINDOW }, LARGE_EMBEDDING_SIZE, F>::random())
    }

    #[inline]
    /// Create new huge model with random parameters.
    pub fn huge() -> Self {
        Self::Huge(GenericLSTMGenerator::<{ HUGE_EMBEDDING_SIZE * HUGE_CONTEXT_WINDOW }, HUGE_EMBEDDING_SIZE, F>::random())
    }

    #[inline]
    /// Create new giant model with random parameters.
    pub fn giant() -> Self {
        Self::Giant(GenericLSTMGenerator::<{ GIANT_EMBEDDING_SIZE * GIANT_CONTEXT_WINDOW }, GIANT_EMBEDDING_SIZE, F>::random())
    }

    /// Get information about the underlying standard generic model.
    pub const fn params(&self) -> SizedModelParams {
        match self {
            Self::Tiny(_) => SizedModelParams {
                embedding_dimensions: TINY_EMBEDDING_SIZE,
                context_window: TINY_CONTEXT_WINDOW,
                parameters: GenericLSTMGenerator::<{ TINY_EMBEDDING_SIZE * TINY_CONTEXT_WINDOW }, TINY_EMBEDDING_SIZE, F>::PARAMS
            },

            Self::Small(_) => SizedModelParams {
                embedding_dimensions: SMALL_EMBEDDING_SIZE,
                context_window: SMALL_CONTEXT_WINDOW,
                parameters: GenericLSTMGenerator::<{ SMALL_EMBEDDING_SIZE * SMALL_CONTEXT_WINDOW }, SMALL_EMBEDDING_SIZE, F>::PARAMS
            },

            Self::Medium(_) => SizedModelParams {
                embedding_dimensions: MEDIUM_EMBEDDING_SIZE,
                context_window: MEDIUM_CONTEXT_WINDOW,
                parameters: GenericLSTMGenerator::<{ MEDIUM_EMBEDDING_SIZE * MEDIUM_CONTEXT_WINDOW }, MEDIUM_EMBEDDING_SIZE, F>::PARAMS
            },

            Self::Large(_) => SizedModelParams {
                embedding_dimensions: LARGE_EMBEDDING_SIZE,
                context_window: LARGE_CONTEXT_WINDOW,
                parameters: GenericLSTMGenerator::<{ LARGE_EMBEDDING_SIZE * LARGE_CONTEXT_WINDOW }, LARGE_EMBEDDING_SIZE, F>::PARAMS
            },

            Self::Huge(_) => SizedModelParams {
                embedding_dimensions: HUGE_EMBEDDING_SIZE,
                context_window: HUGE_CONTEXT_WINDOW,
                parameters: GenericLSTMGenerator::<{ HUGE_EMBEDDING_SIZE * HUGE_CONTEXT_WINDOW }, HUGE_EMBEDDING_SIZE, F>::PARAMS
            },

            Self::Giant(_) => SizedModelParams {
                embedding_dimensions: GIANT_EMBEDDING_SIZE,
                context_window: GIANT_CONTEXT_WINDOW,
                parameters: GenericLSTMGenerator::<{ GIANT_EMBEDDING_SIZE * GIANT_CONTEXT_WINDOW }, GIANT_EMBEDDING_SIZE, F>::PARAMS
            }
        }
    }

    /// Perform forward propagation and return updated long-term memory, short-term memory
    /// and output of the model.
    ///
    /// Note: model is very sensitive. Be accurate with sizes of input slices.
    pub fn forward(&self, long_memory: &[F], short_memory: &[F], input: &[F], device: &impl Device) -> (Vec<F>, Vec<F>, Vec<F>) {
        match self {
            Self::Tiny(model) => {
                let result = model.forward(scale_slice(long_memory), scale_slice(short_memory), scale_slice(input), device);

                (result.0.to_vec(), result.1.to_vec(), result.2.to_vec())
            }

            Self::Small(model) => {
                let result = model.forward(scale_slice(long_memory), scale_slice(short_memory), scale_slice(input), device);

                (result.0.to_vec(), result.1.to_vec(), result.2.to_vec())
            }

            Self::Medium(model) => {
                let result = model.forward(scale_slice(long_memory), scale_slice(short_memory), scale_slice(input), device);

                (result.0.to_vec(), result.1.to_vec(), result.2.to_vec())
            }

            Self::Large(model) => {
                let result = model.forward(scale_slice(long_memory), scale_slice(short_memory), scale_slice(input), device);

                (result.0.to_vec(), result.1.to_vec(), result.2.to_vec())
            }

            Self::Huge(model) => {
                let result = model.forward(scale_slice(long_memory), scale_slice(short_memory), scale_slice(input), device);

                (result.0.to_vec(), result.1.to_vec(), result.2.to_vec())
            }

            Self::Giant(model) => {
                let result = model.forward(scale_slice(long_memory), scale_slice(short_memory), scale_slice(input), device);

                (result.0.to_vec(), result.1.to_vec(), result.2.to_vec())
            }
        }
    }
}

impl<F: Float> From<GenericLSTMGenerator<{ TINY_EMBEDDING_SIZE * TINY_CONTEXT_WINDOW }, TINY_EMBEDDING_SIZE, F>> for SizedModel<F> {
    #[inline(always)]
    fn from(model: GenericLSTMGenerator<{ TINY_EMBEDDING_SIZE * TINY_CONTEXT_WINDOW }, TINY_EMBEDDING_SIZE, F>) -> Self {
        Self::Tiny(model)
    }
}

impl<F: Float> From<GenericLSTMGenerator<{ SMALL_EMBEDDING_SIZE * SMALL_CONTEXT_WINDOW }, SMALL_EMBEDDING_SIZE, F>> for SizedModel<F> {
    #[inline(always)]
    fn from(model: GenericLSTMGenerator<{ SMALL_EMBEDDING_SIZE * SMALL_CONTEXT_WINDOW }, SMALL_EMBEDDING_SIZE, F>) -> Self {
        Self::Small(model)
    }
}

impl<F: Float> From<GenericLSTMGenerator<{ MEDIUM_EMBEDDING_SIZE * MEDIUM_CONTEXT_WINDOW }, MEDIUM_EMBEDDING_SIZE, F>> for SizedModel<F> {
    #[inline(always)]
    fn from(model: GenericLSTMGenerator<{ MEDIUM_EMBEDDING_SIZE * MEDIUM_CONTEXT_WINDOW }, MEDIUM_EMBEDDING_SIZE, F>) -> Self {
        Self::Medium(model)
    }
}

impl<F: Float> From<GenericLSTMGenerator<{ LARGE_EMBEDDING_SIZE * LARGE_CONTEXT_WINDOW }, LARGE_EMBEDDING_SIZE, F>> for SizedModel<F> {
    #[inline(always)]
    fn from(model: GenericLSTMGenerator<{ LARGE_EMBEDDING_SIZE * LARGE_CONTEXT_WINDOW }, LARGE_EMBEDDING_SIZE, F>) -> Self {
        Self::Large(model)
    }
}

impl<F: Float> From<GenericLSTMGenerator<{ HUGE_EMBEDDING_SIZE * HUGE_CONTEXT_WINDOW }, HUGE_EMBEDDING_SIZE, F>> for SizedModel<F> {
    #[inline(always)]
    fn from(model: GenericLSTMGenerator<{ HUGE_EMBEDDING_SIZE * HUGE_CONTEXT_WINDOW }, HUGE_EMBEDDING_SIZE, F>) -> Self {
        Self::Huge(model)
    }
}

impl<F: Float> From<GenericLSTMGenerator<{ GIANT_EMBEDDING_SIZE * GIANT_CONTEXT_WINDOW }, GIANT_EMBEDDING_SIZE, F>> for SizedModel<F> {
    #[inline(always)]
    fn from(model: GenericLSTMGenerator<{ GIANT_EMBEDDING_SIZE * GIANT_CONTEXT_WINDOW }, GIANT_EMBEDDING_SIZE, F>) -> Self {
        Self::Giant(model)
    }
}
