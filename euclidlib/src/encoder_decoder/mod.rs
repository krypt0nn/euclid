use crate::prelude::*;

#[derive(Debug, Clone, PartialEq)]
/// Encoder-decoder model translates input vector into intermediate
/// (encoded) variant with a different size, and learns to represent
/// it back to the proper input. In other words, this model trains
/// to perform `decode(encode(original)) == original` operation.
pub struct EncoderDecoder<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> {
    encoder: Layer<INPUT_SIZE, OUTPUT_SIZE, F>,
    decoder: Layer<OUTPUT_SIZE, INPUT_SIZE, F>
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> EncoderDecoder<INPUT_SIZE, OUTPUT_SIZE, F> {
    /// Amount of parameters in the model.
    pub const PARAMS: usize = (INPUT_SIZE + 1) * OUTPUT_SIZE + (OUTPUT_SIZE + 1) * INPUT_SIZE;

    #[inline]
    /// Create new model with random weights.
    pub fn random() -> Self {
        Self {
            encoder: Layer::linear(),
            decoder: Layer::linear()
        }
    }

    #[inline]
    /// Resize model by truncating neurons and weights of the neurons layers or repeating them.
    pub fn resize<const NEW_INPUT_SIZE: usize, const NEW_OUTPUT_SIZE: usize>(&self) -> EncoderDecoder<NEW_INPUT_SIZE, NEW_OUTPUT_SIZE, F> {
        EncoderDecoder {
            encoder: self.encoder.resize(),
            decoder: self.decoder.resize()
        }
    }

    #[inline]
    /// Encode given input.
    pub fn encode(&self, input: impl IntoHeapArray<F, INPUT_SIZE>) -> Box<[F; OUTPUT_SIZE]> {
        self.encoder.forward(input)
    }

    #[inline]
    /// Decode given output.
    pub fn decode(&self, output: impl IntoHeapArray<F, OUTPUT_SIZE>) -> Box<[F; INPUT_SIZE]> {
        self.decoder.forward(output)
    }

    /// Train the model to perform `encode(input) = output` conversion.
    ///
    /// Returns gradients from the encoder layer.
    pub fn train_encode(
        &mut self,
        input: impl IntoHeapArray<F, INPUT_SIZE>,
        output: impl IntoHeapArray<F, OUTPUT_SIZE>,
        policy: &mut BackpropagationSnapshot<{ (INPUT_SIZE + 1) * OUTPUT_SIZE }, F>
    ) -> Box<[F; INPUT_SIZE]> {
        self.encoder.backward(input, output, policy)
    }

    /// Train the model to perform `decode(output) = input` conversion.
    ///
    /// Returns gradients from the decoder layer.
    pub fn train_decode(
        &mut self,
        output: impl IntoHeapArray<F, OUTPUT_SIZE>,
        input: impl IntoHeapArray<F, INPUT_SIZE>,
        policy: &mut BackpropagationSnapshot<{ (OUTPUT_SIZE + 1) * INPUT_SIZE }, F>
    ) -> Box<[F; OUTPUT_SIZE]> {
        self.decoder.backward(output, input, policy)
    }

    #[allow(unused_braces)]
    /// Train the model to perform `decode(encode(input)) == input` conversion.
    pub fn train(
        &mut self,
        input: impl IntoHeapArray<F, INPUT_SIZE>,
        policy: &mut BackpropagationSnapshot<{ Self::PARAMS }, F>
    ) where
        [(); { (OUTPUT_SIZE + 1) * INPUT_SIZE }]: Sized,
        [(); { (INPUT_SIZE + 1) * OUTPUT_SIZE }]: Sized
    {
        let input = unsafe { input.into_heap_array() };

        let encoded = self.encoder.forward(&input);

        let gradients = policy.window::<{ (OUTPUT_SIZE + 1) * INPUT_SIZE }, _>(0, |mut policy| {
            self.decoder.backward(&encoded, &input, &mut policy)
        });

        policy.window::<{ (INPUT_SIZE + 1) * OUTPUT_SIZE }, _>((OUTPUT_SIZE + 1) * INPUT_SIZE, |mut policy| {
            self.encoder.backward_propagated(&input, &gradients, &mut policy);
        });
    }

    #[inline]
    /// Calculate encoder loss.
    pub fn encoder_loss(
        &self,
        actual_output: impl IntoHeapArray<F, OUTPUT_SIZE>,
        expected_output: impl IntoHeapArray<F, OUTPUT_SIZE>
    ) -> F {
        self.encoder.loss(actual_output, expected_output)
    }

    #[inline]
    /// Calculate decoder loss.
    pub fn decoder_loss(
        &self,
        actual_input: impl IntoHeapArray<F, INPUT_SIZE>,
        expected_input: impl IntoHeapArray<F, INPUT_SIZE>
    ) -> F {
        self.decoder.loss(actual_input, expected_input)
    }

    #[inline]
    /// Calculate total encoder-decoder loss.
    pub fn loss(&self, input: impl IntoHeapArray<F, INPUT_SIZE>) -> F {
        let input = unsafe { input.into_heap_array() };

        let actual = self.decode(self.encode(&input));

        self.decoder_loss(input, actual)
    }
}

#[test]
/// Test identity encoder-decoder.
fn test_encoder_decoder() {
    // Prepare encoder-decoder model.
    let mut model = EncoderDecoder::<2, 2, f64>::random();

    // Prepare backpropagation policy.
    let mut backpropagation = Backpropagation::default();

    // Train the model on given samples.
    let examples = [
        [0.0, 0.0],
        [0.1, 0.0],
        [0.6, 0.3],
        [0.0, 0.8],
        [0.5, 0.5],
        [1.0, 1.0]
    ];

    for _ in 0..1000 {
        for example in examples {
            backpropagation.timestep(|mut policy| {
                model.train(example, &mut policy);
            })
        }
    }

    // Validate the model.
    assert!(model.loss([0.71, 0.13]) < 0.3);
}
