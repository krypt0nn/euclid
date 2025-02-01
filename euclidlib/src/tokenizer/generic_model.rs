use crate::prelude::*;

#[derive(Debug, Clone, PartialEq)]
/// Generic word2vec-like model.
///
/// This neural network allows you to encode given tokens
/// into multi-dimentional vectors (word embeddings) where each
/// dimension has some meaning and allows other natural language
/// models to use this information for much better training.
///
/// Consist of encoder-decoder model which converts tokens to embeddings,
/// and embeddings summator which uses `CONTEXT_SIZE / EMBEDDING_SIZE`
/// embeddings to find new embedding using given ones as a context.
///
/// Combined together embeddings summator allows word embeddings to
/// store semantic information about words in real text corpuses and
/// be able to encode and decode separate tokens precisely.
///
/// ! `CONTEXT_SIZE = N * EMBEDDING_SIZE`. This is a rust generics limitation.
pub struct GenericModel<const TOKENS_NUM: usize, const EMBEDDING_SIZE: usize, const CONTEXT_SIZE: usize, F: Float> {
    encoder_decoder: EncoderDecoder<TOKENS_NUM, EMBEDDING_SIZE, F>,
    embeddings_context: Layer<CONTEXT_SIZE, EMBEDDING_SIZE, F>
}

impl<
    const TOKENS_NUM: usize,
    const EMBEDDING_SIZE: usize,
    const CONTEXT_SIZE: usize,
    F: Float
> GenericModel<TOKENS_NUM, EMBEDDING_SIZE, CONTEXT_SIZE, F> {
    /// Amount of parameters in the current generic model.
    pub const PARAMS: usize = EncoderDecoder::<TOKENS_NUM, EMBEDDING_SIZE, F>::MODEL_PARAMS + Layer::<CONTEXT_SIZE, EMBEDDING_SIZE, F>::PARAMS;

    #[inline]
    /// Create new word embeddings model with random weights.
    pub fn random() -> Self {
        Self {
            encoder_decoder: EncoderDecoder::random(),
            embeddings_context: Layer::linear()
        }
    }

    #[inline]
    /// Resize model by truncating neurons and weights of the neurons layers or repeating them.
    pub fn resize<
        const NEW_TOKENS_NUM: usize,
        const NEW_EMBEDDING_SIZE: usize,
        const NEW_CONTEXT_SIZE: usize
    >(&self) -> GenericModel<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE, NEW_CONTEXT_SIZE, F> {
        GenericModel {
            encoder_decoder: self.encoder_decoder.resize(),
            embeddings_context: self.embeddings_context.resize()
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
    pub fn encode(&self, token: usize) -> Box<[F; EMBEDDING_SIZE]> {
        self.encoder_decoder.encode(Self::get_one_hot_embedding(&[token]))
    }

    /// Decode given embedding vector to the token.
    pub fn decode(&self, embedding: impl IntoHeapArray<F, EMBEDDING_SIZE>) -> usize {
        let probs = self.encoder_decoder.decode(embedding);

        let mut max = (0, 0.0);

        for i in 0..TOKENS_NUM {
            let prob = probs[i].as_f32();

            if prob > max.1 {
                max = (i, prob);
            }
        }

        max.0
    }

    /// Calculate cosine distance between two word embeddings.
    ///
    /// Closer two words to each other - lower the output value.
    pub fn distance(&self, word_1: &[F; EMBEDDING_SIZE], word_2: &[F; EMBEDDING_SIZE]) -> f64 {
        let mut distance = 0.0;
        let mut len_1 = 0.0;
        let mut len_2 = 0.0;

        for i in 0..EMBEDDING_SIZE {
            distance += word_1[i].as_f64() * word_2[i].as_f64();

            len_1 += word_1[i].as_f64().powi(2);
            len_2 += word_2[i].as_f64().powi(2);
        }

        distance / (len_1.sqrt() * len_2.sqrt())
    }

    #[allow(unused_braces)]
    /// Train the model once on given tokens slice with given radius and policy.
    /// Radius specifies amount of tokens to left and right from the target one.
    /// For example, in tokens `[0, 1, 2, 3, 4]` and `radius = 2` for target token
    /// `2` the input slice will be `[0, 1, 3, 4]`. In other words, you train
    /// the model to predict token sitting in the middle of `radius * 2` tokens.
    pub fn train(
        &mut self,
        tokens: &[usize],
        policy: &mut BackpropagationSnapshot<'_, { Self::PARAMS }, F>
    )
    where
        [(); { Layer::<EMBEDDING_SIZE, TOKENS_NUM, F>::PARAMS }]: Sized,
        [(); { Layer::<CONTEXT_SIZE, EMBEDDING_SIZE, F>::PARAMS }]: Sized,
        [(); { Layer::<TOKENS_NUM, EMBEDDING_SIZE, F>::PARAMS }]: Sized,
        [(); { Neuron::<EMBEDDING_SIZE, F>::PARAMS }]: Sized,
        [(); { Neuron::<CONTEXT_SIZE, F>::PARAMS }]: Sized,
        [(); { Neuron::<TOKENS_NUM, F>::PARAMS }]: Sized
    {
        let n = tokens.len();

        // Prepare heap buffers.
        let mut context = unsafe {
            alloc_fixed_heap_array_with::<F, CONTEXT_SIZE>(F::ZERO)
                .expect("Failed to allocate memory for word embeddings context")
        };

        let mut one_hot = unsafe {
            alloc_fixed_heap_array_with::<F, TOKENS_NUM>(F::ZERO)
                .expect("Failed to allocate memory for word embeddings one-hot encoding")
        };

        let mut embedding = unsafe {
            alloc_fixed_heap_array_with::<F, EMBEDDING_SIZE>(F::ZERO)
                .expect("Failed to allocate memory for word embeddings")
        };

        // Amount of tokens within the context window.
        let context_tokens = CONTEXT_SIZE / EMBEDDING_SIZE;

        // For `context_embeddings = 3` we get `radius_r = 1` and `radius_l = 2`
        // so we use more previous tokens for the context.
        let radius_r = context_tokens / 2;
        let radius_l = context_tokens - radius_r;

        // For each token in the input sequence.
        for i in radius_l..n - radius_r {
            // 1. Get `context_embeddings` embeddings around the token.
            let mut k = 0;

            #[allow(clippy::needless_range_loop)]
            for j in i - radius_l..i + radius_r {
                if j != i {
                    let embedding = self.encode(tokens[j]);

                    context[k * EMBEDDING_SIZE..(k + 1) * EMBEDDING_SIZE].copy_from_slice(embedding.as_slice());

                    k += 1;
                }
            }

            // 2. Predict token from its context.
            let predicted_embedding = self.embeddings_context.forward(&context);

            // 3. Train decoder to assign `predicted_embedding` for `tokens[i]`.
            one_hot[tokens[i]] = F::ONE;

            let decoder_gradients = policy.window::<{ Layer::<EMBEDDING_SIZE, TOKENS_NUM, F>::PARAMS }, _>(0, |mut policy| {
                self.encoder_decoder.train_decode(&predicted_embedding, &one_hot, &mut policy)
            });

            one_hot[tokens[i]] = F::ZERO;

            // 4. Train context summator using decoder's gradients.
            let context_gradients = policy.window::<{ Layer::<CONTEXT_SIZE, EMBEDDING_SIZE, F>::PARAMS }, _>(Layer::<EMBEDDING_SIZE, TOKENS_NUM, F>::PARAMS, |mut policy| {
                self.embeddings_context.backward_propagated(&context, &decoder_gradients, &mut policy)
            });

            // 5. Train encoder to assign `context[..]` for `tokens[j]` using context's gradients.
            k = 0;

            let policy_offset = Layer::<EMBEDDING_SIZE, TOKENS_NUM, F>::PARAMS + Layer::<CONTEXT_SIZE, EMBEDDING_SIZE, F>::PARAMS;

            for j in i - radius_l..i + radius_r {
                if j != i {
                    one_hot[tokens[j]] = F::ONE;

                    policy.window::<{ Layer::<TOKENS_NUM, EMBEDDING_SIZE, F>::PARAMS }, _>(policy_offset + k * Layer::<TOKENS_NUM, EMBEDDING_SIZE, F>::PARAMS, |mut policy| {
                        embedding.copy_from_slice(&context_gradients[k * EMBEDDING_SIZE..(k + 1) * EMBEDDING_SIZE]);

                        self.encoder_decoder.encoder.backward_propagated(&one_hot, &embedding, &mut policy);
                    });

                    one_hot[tokens[j]] = F::ZERO;

                    k += 1;
                }
            }
        }
    }

    /// Calculate loss between expected token and an actual prediction of the model.
    pub fn loss(&self, input_tokens: &[usize], expected_output: usize) -> F {
        let mut context_embeddings = unsafe {
            alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for word embeddings model context")
        };

        let mut outputs_one_hot = unsafe {
            alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for word embeddings model output")
        };

        for k in 0..input_tokens.len() {
            context_embeddings[
                k * EMBEDDING_SIZE..
                (k + 1) * EMBEDDING_SIZE
            ].copy_from_slice(self.encode(input_tokens[k]).as_slice());
        }

        let predicted_embedding = self.embeddings_context.forward(context_embeddings);
        let predicted_tokens = self.encoder_decoder.decode(predicted_embedding);

        outputs_one_hot[expected_output] = F::ONE;

        self.encoder_decoder.decoder_loss(predicted_tokens, outputs_one_hot)
    }
}
