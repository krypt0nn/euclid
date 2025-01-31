use crate::prelude::*;

#[derive(Debug, Clone, PartialEq)]
/// Generic word2vec-like model.
///
/// This 2-layer neural network allows you to encode given tokens
/// into multi-dimentional vectors (word embeddings) where each
/// dimension has some meaning and allows other natural language
/// models to use this information for much better training.
pub struct GenericModel<const TOKENS_NUM: usize, const EMBEDDING_SIZE: usize, F: Float> {
    input_layer: Layer<TOKENS_NUM, EMBEDDING_SIZE, F>,
    output_layer: Layer<EMBEDDING_SIZE, TOKENS_NUM, F>
}

impl<const TOKENS_NUM: usize, const EMBEDDING_SIZE: usize, F: Float> GenericModel<TOKENS_NUM, EMBEDDING_SIZE, F> {
    /// Amount of parameters in the current generic model.
    pub const PARAMS: usize = (TOKENS_NUM + 1) * EMBEDDING_SIZE + (EMBEDDING_SIZE + 1) * TOKENS_NUM;

    #[inline]
    /// Create new word embeddings model with random weights.
    pub fn random() -> Self {
        Self {
            input_layer: Layer::sigmoid(),
            output_layer: Layer::sigmoid()
        }
    }

    #[inline]
    /// Resize model by truncating neurons and weights of the neurons layers or repeating them.
    pub fn resize<const NEW_TOKENS_NUM: usize, const NEW_EMBEDDING_SIZE: usize>(&self) -> GenericModel<NEW_TOKENS_NUM, NEW_EMBEDDING_SIZE, F> {
        GenericModel {
            input_layer: self.input_layer.resize(),
            output_layer: self.output_layer.resize()
        }
    }

    /// Get given token embedding.
    pub fn get_embedding(&self, token: usize) -> Box<[F; EMBEDDING_SIZE]> {
        let mut input = unsafe {
            alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for word embeddings model input")
        };

        input[token] = F::ONE;

        self.input_layer.forward(input)
    }

    /// Find output token with the greatest probability of occurence
    /// among the given group of input tokens.
    pub fn predict(&self, input_tokens: &[usize]) -> usize {
        let mut input = unsafe {
            alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for word embeddings model input")
        };

        for token in input_tokens {
            input[*token] = F::ONE;
        }

        let hidden_input = self.input_layer.forward(&input);
        let probabilities = self.output_layer.forward(&hidden_input);

        probabilities.into_iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                if a.as_f32() > b.as_f32() {
                    std::cmp::Ordering::Greater
                } else if a == b {
                    std::cmp::Ordering::Equal
                } else {
                    std::cmp::Ordering::Less
                }
            })
            .map(|(i, _)| i)
            .unwrap_or_default()
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
    pub fn train<const RADIUS: usize>(
        &mut self,
        tokens: &[usize],
        policy: &mut BackpropagationSnapshot<'_, { (TOKENS_NUM + 1) * EMBEDDING_SIZE + (EMBEDDING_SIZE + 1) * TOKENS_NUM }, F>
    )
    where
        [(); { (TOKENS_NUM + 1) * EMBEDDING_SIZE + (EMBEDDING_SIZE + 1) * TOKENS_NUM }]: Sized
    {
        let n = tokens.len();

        // Prepare input and output arrays outside of the loop for less allocations.
        let mut input = unsafe {
            alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for word embeddings model input")
        };

        let mut output = unsafe {
            alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for word embeddings model output")
        };

        for i in RADIUS..n - RADIUS {
            // Set ones to input and output arrays where it's needed.
            for j in i - RADIUS..i + RADIUS {
                input[tokens[j]] = F::ONE;
            }

            input[tokens[i]]  = F::ZERO;
            output[tokens[i]] = F::ONE;

            // Train on these arrays.
            let forward = self.input_layer.forward(&input);

            let gradients = policy.window(0, |mut policy| {
                // TOKENS_NUM output neurons with EMBEDDING_SIZE weights + 1 bias each.
                self.output_layer.backward(&forward, &output, &mut policy)
            });

            policy.window((EMBEDDING_SIZE + 1) * TOKENS_NUM, |mut policy| {
                self.input_layer.backward_propagated(&input, &gradients, &mut policy);
            });

            // Zero input and output arrays back.
            for j in i - RADIUS..i + RADIUS {
                input[tokens[j]] = F::ONE;
            }

            output[tokens[i]] = F::ZERO;
        }
    }

    /// Calculate loss between expected token and an actual prediction of the model.
    pub fn loss(&self, input_tokens: &[usize], expected_output: usize) -> F {
        let mut input = unsafe {
            alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for word embeddings model input")
        };

        let mut output = unsafe {
            alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for word embeddings model output")
        };

        for token in input_tokens {
            input[*token] = F::ONE;
        }

        output[expected_output] = F::ONE;

        let hidden_input = self.input_layer.forward(input);
        let probabilities = self.output_layer.forward(&hidden_input);

        self.output_layer.loss(&probabilities, &output)
    }
}
