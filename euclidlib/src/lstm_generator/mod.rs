use crate::prelude::*;

/// Recurrent generator model using Long Short-Term Memory cells.
pub struct LSTMGenerator<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> {
    pub lstm_layer: LSTMLayer<INPUT_SIZE, F>,
    pub output_layer: Layer<INPUT_SIZE, OUTPUT_SIZE, F>
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> LSTMGenerator<INPUT_SIZE, OUTPUT_SIZE, F> {
    /// Amount of parameters of the model.
    pub const PARAMS: usize = LSTMLayer::<INPUT_SIZE, F>::PARAMS + Layer::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS;

    /// Create new LSTM generator with random parameters.
    pub fn random() -> Self {
        Self {
            lstm_layer: LSTMLayer::random(),
            output_layer: Layer::linear()
        }
    }

    #[allow(clippy::type_complexity)]
    /// Calculate output from the given memory values and inputs
    /// (perform forward propagation).
    ///
    /// This method will return updated long-term, short-term memory
    /// and output of the model.
    pub fn forward(
        &self,
        long_memory: impl IntoHeapArray<F, INPUT_SIZE>,
        short_memory: impl IntoHeapArray<F, INPUT_SIZE>,
        inputs: impl IntoHeapArray<F, INPUT_SIZE>,
        device: &impl Device
    ) -> (Box<[F; INPUT_SIZE]>, Box<[F; INPUT_SIZE]>, Box<[F; OUTPUT_SIZE]>) {
        let (long_memory, short_memory) = self.lstm_layer.forward(long_memory, short_memory, inputs);

        let output = self.output_layer.forward(&short_memory, device);

        (long_memory, short_memory, output)
    }

    #[allow(unused_braces)]
    /// Train the model to predict some output from the preceding input
    /// in the provided sequence.
    pub fn train(
        &mut self,
        sequence: &[F],
        step: usize,
        policy: &mut BackpropagationSnapshot<{ LSTMGenerator::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS }, F>,
        device: &mut impl Device
    )
    where
        [(); { LSTMGenerator::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS }]: Sized,
        [(); { LSTMLayer::<INPUT_SIZE, F>::PARAMS }]: Sized,
        [(); { LSTMCell::<F>::PARAMS }]: Sized,
        [(); { Layer::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS }]: Sized,
        [(); { Neuron::<INPUT_SIZE, F>::PARAMS }]: Sized,
        [(); { Neuron::<2, F>::PARAMS }]: Sized,

        [(); INPUT_SIZE]: Sized,
        [(); OUTPUT_SIZE]: Sized
    {
        // Skip training if sequence is too short.
        if sequence.len() < INPUT_SIZE + OUTPUT_SIZE {
            return;
        }

        let n = sequence.len() - INPUT_SIZE - OUTPUT_SIZE;

        let mut i = 0;

        unsafe {
            let mut input = alloc_fixed_heap_array_with::<F, INPUT_SIZE>(F::ZERO)
                .expect("Failed to allocate memory for LSTM generator input slice");

            let mut output = alloc_fixed_heap_array_with::<F, OUTPUT_SIZE>(F::ZERO)
                .expect("Failed to allocate memory for LSTM generator output slice");

            let mut long_memory = alloc_fixed_heap_array_with::<F, INPUT_SIZE>(F::ZERO)
                .expect("Failed to allocate memory for LSTM generator long-term memory");

            let mut short_memory = alloc_fixed_heap_array_with::<F, INPUT_SIZE>(F::ZERO)
                .expect("Failed to allocate memory for LSTM generator short-term memory");

            let mut gradients = alloc_fixed_heap_array_with::<F, OUTPUT_SIZE>(F::ZERO)
                .expect("Failed to allocate memory for LSTM generator gradients");

            let mut forward = Vec::with_capacity(n / step);

            forward.push((
                long_memory.clone(),
                short_memory.clone(),
                input.clone(),
                output.clone(), // expected output
                output.clone()  // actual output
            ));

            // Forward pass.
            while i < n {
                input[..INPUT_SIZE].copy_from_slice(&sequence[i..i + INPUT_SIZE]);
                output[..OUTPUT_SIZE].copy_from_slice(&sequence[i + INPUT_SIZE..i + INPUT_SIZE + OUTPUT_SIZE]);

                (long_memory, short_memory) = self.lstm_layer.forward(&long_memory, &short_memory, &input);

                let actual_output = self.output_layer.forward(&short_memory, device);

                forward.push((
                    long_memory.clone(),
                    short_memory.clone(),
                    input.clone(),
                    output.clone(),
                    actual_output
                ));

                i += step;
            }

            // Prepare gradients for backward passes.
            let n = forward.len();

            let (_, short_memory, input, expected_output, _) = &forward[n - 1];

            let mut output_gradients = policy.window::<{ Layer::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS }, _>(0, |mut policy| {
                self.output_layer.backward(short_memory, expected_output, &mut policy, device)
            });

            let (long_memory, short_memory, _, _, _) = &forward[n - 2];

            let mut lstm_gradients = policy.window({ Layer::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS }, |mut policy| {
                self.lstm_layer.backward_propagated(long_memory, short_memory, input, &output_gradients, &mut policy)
            });

            // Backward propagation.
            if n > 2 {
                let gradients_div = F::from_float(INPUT_SIZE as f32 / OUTPUT_SIZE as f32);

                for i in (1..n - 3).rev() {
                    let (_, short_memory, input, _, _) = &forward[i];

                    gradients.copy_from_slice(&lstm_gradients[..OUTPUT_SIZE]);

                    let mut j = OUTPUT_SIZE;

                    while j < INPUT_SIZE {
                        for k in 0..OUTPUT_SIZE {
                            gradients[k] += lstm_gradients[j + k];
                        }

                        j += OUTPUT_SIZE;
                    }

                    for k in 0..OUTPUT_SIZE {
                        gradients[k] *= gradients_div;
                    }

                    output_gradients = policy.window::<{ Layer::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS }, _>(0, |mut policy| {
                        self.output_layer.backward_propagated(short_memory, &gradients, &mut policy, device)
                    });

                    let (long_memory, short_memory, _, _, _) = &forward[i - 1];

                    lstm_gradients = policy.window({ Layer::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS }, |mut policy| {
                        self.lstm_layer.backward_propagated(long_memory, short_memory, input, &output_gradients, &mut policy)
                    });
                }
            }
        }
    }
}
