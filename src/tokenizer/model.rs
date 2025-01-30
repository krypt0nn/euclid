use crate::prelude::*;

#[derive(Debug, Clone, PartialEq)]
/// Generic word2vec-like model.
///
/// This 2-layer neural network allows you to encode given tokens
/// into multi-dimentional vectors (word embeddings) where each
/// dimension has some meaning and allows other natural language
/// models to use this information for much better training.
pub struct Model<const TOKENS_NUM: usize, const EMBEDDING_SIZE: usize, F: Float> {
    input_layer: Layer<TOKENS_NUM, EMBEDDING_SIZE, F>,
    output_layer: Layer<EMBEDDING_SIZE, TOKENS_NUM, F>
}

impl<const TOKENS_NUM: usize, const EMBEDDING_SIZE: usize, F: Float> Model<TOKENS_NUM, EMBEDDING_SIZE, F> {
    #[inline]
    /// Create new word embeddings model with random weights.
    pub fn random() -> Self {
        Self {
            input_layer: Layer::new(sigmoid, sigmoid_derivative, quadratic_error, quadratic_error_derivative),
            output_layer: Layer::new(sigmoid, sigmoid_derivative, cross_entropy, cross_entropy_derivative)
        }
    }

    /// Get given token embedding.
    pub fn get_embedding(&self, token: usize) -> [F; EMBEDDING_SIZE] {
        let mut input = [F::ZERO; TOKENS_NUM];

        input[token] = F::ONE;

        self.input_layer.forward(&input)
    }

    /// Find output token with the greatest probability of occurence
    /// among the given group of input tokens.
    pub fn predict(&self, input_tokens: &[usize]) -> usize {
        let mut input = [F::ZERO; TOKENS_NUM];

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

        for i in RADIUS..n - RADIUS {
            let mut input = [F::ZERO; TOKENS_NUM];
            let mut output = [F::ZERO; TOKENS_NUM];

            #[allow(clippy::needless_range_loop)]
            for j in i - RADIUS..i + RADIUS {
                input[tokens[j]] = F::ONE;
            }

            input[tokens[i]]  = F::ZERO;
            output[tokens[i]] = F::ONE;

            let forward = self.input_layer.forward(&input);

            let gradients = policy.window(0, |mut policy| {
                // TOKENS_NUM output neurons with EMBEDDING_SIZE weights + 1 bias each.
                self.output_layer.backward(&forward, &output, &mut policy)
            });

            policy.window((EMBEDDING_SIZE + 1) * TOKENS_NUM, |mut policy| {
                self.input_layer.backward_propagated(&input, &gradients, &mut policy);
            });
        }
    }

    /// Calculate loss between expected token and an actual prediction of the model.
    pub fn loss(&self, input_tokens: &[usize], expected_output: usize) -> F {
        let mut input = [F::ZERO; TOKENS_NUM];
        let mut output = [F::ZERO; TOKENS_NUM];

        for token in input_tokens {
            input[*token] = F::ONE;
        }

        output[expected_output] = F::ONE;

        let hidden_input = self.input_layer.forward(&input);
        let probabilities = self.output_layer.forward(&hidden_input);

        self.output_layer.loss(&probabilities, &output)
    }
}
