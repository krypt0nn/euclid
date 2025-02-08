use crate::prelude::*;

pub const TINY_CONTEXT_WINDOW: usize   = 8;
pub const SMALL_CONTEXT_WINDOW: usize  = 16;
pub const MEDIUM_CONTEXT_WINDOW: usize = 24;
pub const LARGE_CONTEXT_WINDOW: usize  = 32;
pub const HUGE_CONTEXT_WINDOW: usize   = 48;
pub const GIANT_CONTEXT_WINDOW: usize  = 64;

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

    // TODO
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
