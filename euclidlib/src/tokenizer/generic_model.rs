use crate::prelude::*;

#[derive(Debug, Clone, PartialEq)]
/// Generic word2vec-like model.
///
/// This neural network allows you to encode given tokens
/// into multi-dimentional vectors (word embeddings) where each
/// dimension has some meaning and allows other natural language
/// models to use this information for much better training.
pub struct GenericModel<const TOKENS_NUM: usize, const EMBEDDING_SIZE: usize, F: Float> {
    pub encoder_decoder: EncoderDecoder<TOKENS_NUM, EMBEDDING_SIZE, F>
}

impl<const TOKENS_NUM: usize, const EMBEDDING_SIZE: usize, F: Float> GenericModel<TOKENS_NUM, EMBEDDING_SIZE, F> {
    /// Amount of parameters in the current generic model.
    pub const PARAMS: usize = EncoderDecoder::<TOKENS_NUM, EMBEDDING_SIZE, F>::MODEL_PARAMS;

    #[inline]
    /// Create new word embeddings model with random weights.
    pub fn random() -> Self {
        Self {
            encoder_decoder: EncoderDecoder::random()
        }
    }

    #[inline]
    /// Resize model by truncating neurons and weights of the neurons layers or repeating them.
    pub fn resize<
        const NEW_TOKENS_NUM: usize,
        const NEW_EMBEDDING_SIZE: usize
    >(&self) -> GenericModel<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE, F> {
        GenericModel {
            encoder_decoder: self.encoder_decoder.resize()
        }
    }

    /// Get one-hot tokens embeddings.
    pub fn get_one_hot_embedding(tokens: &[usize]) -> Box<[F; TOKENS_NUM]> {
        let mut embedding = unsafe {
            alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for one-hot words embedding")
        };

        for token in tokens {
            embedding[*token] = F::ONE;
        }

        embedding
    }

    /// Encode given token to the embedding vector.
    pub fn encode(&self, token: usize, device: &impl Device) -> Box<[F; EMBEDDING_SIZE]> {
        self.encoder_decoder.encode(Self::get_one_hot_embedding(&[token]), device)
    }

    /// Decode given embedding vector to the token.
    pub fn decode(&self, embedding: impl IntoHeapArray<F, EMBEDDING_SIZE>, device: &impl Device) -> usize {
        let mut tokens_one_hot = unsafe {
            alloc_fixed_heap_array_with::<F, TOKENS_NUM>(F::ZERO)
                .expect("Failed to allocate memory for words one-hot embedding")
        };

        let target_embedding = unsafe {
            embedding.into_heap_array()
        };

        let mut closest_token = (0, f64::MIN);

        for token in 0..TOKENS_NUM {
            tokens_one_hot[token] = F::ONE;

            let embedding = self.encoder_decoder.encode(&tokens_one_hot, device);

            let similarity = cosine_similarity::<EMBEDDING_SIZE, F>(embedding.as_slice(), target_embedding.as_slice());

            if similarity >= 1.0 {
                return token;
            }

            if similarity > closest_token.1 {
                closest_token = (token, similarity);
            }

            tokens_one_hot[token] = F::ZERO;
        }

        closest_token.0
    }

    #[allow(unused_braces)]
    /// Train the model once on given tokens slice with given radius and policy.
    /// Radius specifies amount of tokens to left and right from the target one.
    /// For example, in tokens `[0, 1, 2, 3, 4]` and `radius = 2` for target token
    /// `2` the input slice will be `[0, 1, 3, 4]`. In other words, you train
    /// the model to predict token sitting in the middle of `radius * 2` tokens.
    ///
    /// Internally this method runs `train_skippable` with callback that always returns false.
    pub fn train<const CONTEXT_RADIUS: usize>(
        &mut self,
        tokens: &[usize],
        policy: &mut BackpropagationSnapshot<{ Self::PARAMS }, F>,
        device: &mut impl Device
    )
    where
        [(); { Layer::<EMBEDDING_SIZE, TOKENS_NUM, F>::PARAMS }]: Sized,
        [(); { Layer::<TOKENS_NUM, EMBEDDING_SIZE, F>::PARAMS }]: Sized,
        [(); { Neuron::<EMBEDDING_SIZE, F>::PARAMS }]: Sized,
        [(); { Neuron::<TOKENS_NUM, F>::PARAMS }]: Sized
    {
        self.train_skippable::<CONTEXT_RADIUS>(tokens, |_| false, policy, device);
    }

    #[allow(unused_braces)]
    /// Use `skip_token` callback to skip training for specific tokens.
    /// Can be used to skip non-frequent tokens to improve performance.
    pub fn train_skippable<const CONTEXT_RADIUS: usize>(
        &mut self,
        tokens: &[usize],
        skip_token: impl Fn(usize) -> bool,
        policy: &mut BackpropagationSnapshot<{ Self::PARAMS }, F>,
        device: &mut impl Device
    )
    where
        [(); { Layer::<EMBEDDING_SIZE, TOKENS_NUM, F>::PARAMS }]: Sized,
        [(); { Layer::<TOKENS_NUM, EMBEDDING_SIZE, F>::PARAMS }]: Sized,
        [(); { Neuron::<EMBEDDING_SIZE, F>::PARAMS }]: Sized,
        [(); { Neuron::<TOKENS_NUM, F>::PARAMS }]: Sized
    {
        let n = tokens.len();

        // Prepare heap buffer.
        let mut tokens_one_hot = unsafe {
            alloc_fixed_heap_array_with::<F, TOKENS_NUM>(F::ZERO)
                .expect("Failed to allocate memory for words one-hot embedding")
        };

        // For each token in the input sequence.
        for i in CONTEXT_RADIUS..n - CONTEXT_RADIUS {
            if !skip_token(tokens[i]) {
                // 1. Prepare context tokens vector.
                #[allow(clippy::needless_range_loop)]
                for j in i - CONTEXT_RADIUS..i + CONTEXT_RADIUS {
                    if j != i {
                        tokens_one_hot[tokens[j]] = F::ONE;
                    }
                }

                // 2. Calculate embedding for context tokens.
                let embedding = self.encoder_decoder.encode(&tokens_one_hot, device);

                // 3. Prepare one-hot encoding for target token.
                #[allow(clippy::needless_range_loop)]
                for j in i - CONTEXT_RADIUS..i + CONTEXT_RADIUS {
                    tokens_one_hot[tokens[j]] = F::ZERO;
                }

                tokens_one_hot[tokens[i]] = F::ONE;

                // 4. Train decoder to predict the target token.
                let gradients = policy.window::<{ Layer::<EMBEDDING_SIZE, TOKENS_NUM, F>::PARAMS }, _>(0, |mut policy| {
                    self.encoder_decoder.train_decoder(&embedding, &tokens_one_hot, &mut policy, device)
                });

                // 5. Prepare context tokens vector again.
                #[allow(clippy::needless_range_loop)]
                for j in i - CONTEXT_RADIUS..i + CONTEXT_RADIUS {
                    tokens_one_hot[tokens[j]] = F::ONE;
                }

                tokens_one_hot[tokens[i]] = F::ZERO;

                // 6. Backpropagate encoder with decoder's gradients.
                policy.window::<{ Layer::<TOKENS_NUM, EMBEDDING_SIZE, F>::PARAMS }, _>(Layer::<EMBEDDING_SIZE, TOKENS_NUM, F>::PARAMS, |mut policy| {
                    self.encoder_decoder.encoder.backward_propagated(&tokens_one_hot, &gradients, &mut policy, device);
                });

                // 7. Restore one-hot embedding buffer.
                #[allow(clippy::needless_range_loop)]
                for j in i - CONTEXT_RADIUS..i + CONTEXT_RADIUS {
                    tokens_one_hot[tokens[j]] = F::ZERO;
                }
            }
        }
    }

    /// Calculate loss between expected token and an actual prediction of the model.
    pub fn loss(&self, input_tokens: &[usize], expected_output: usize, device: &impl Device) -> F {
        let encoded = self.encoder_decoder.encode(Self::get_one_hot_embedding(input_tokens), device);
        let probabilities = self.encoder_decoder.decode(encoded, device);

        self.encoder_decoder.decoder_loss(probabilities, Self::get_one_hot_embedding(&[expected_output]))
    }
}
