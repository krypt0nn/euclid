use crate::prelude::*;

/// Generic `LSTMCell` type with f32 float type.
pub type LSTMCell32 = LSTMCell<f32>;

/// Generic `LSTMCell` type with f64 float type.
pub type LSTMCell64 = LSTMCell<f64>;

#[derive(Debug, Clone, PartialEq)]
/// Long Short-Term Memory cell.
///
/// This cell has 3 inputs and 2 outputs:
///
/// - Long-term memory input;
/// - Short-term memory input;
/// - Input value itself.
///
/// Cell uses all three numbers to produce:
///
/// - Updated long-term memory;
/// - Updated short-term memory (also the output from the input).
///
/// LSTM cells are good when you want your model to remember
/// previous information and use it to make forecasts.
///
/// Note: this is not a "classic" LSTM cell implementation but
/// its similar alternative which uses neurons with biases.
pub struct LSTMCell<F: Float> {
    // Stage 1. Forget previous long-term memory.
    pub long_memory_forget_value: Neuron<2, F>, // (short_memory, input) -> long_memory_forget_value
    pub long_memory_forget_gate: Neuron<2, F>,  // (long_memory, long_memory_forget_value) -> new_long_memory

    // Stage 2. Remember new long-term information.
    pub long_memory_remember_value: Neuron<2, F>, // (short_memory, input) -> long_memory_remember_value
    pub long_memory_remember_gate: Neuron<2, F>,  // (long_memory, long_memory_remember_value) -> new_long_memory

    // Stage 3. Produce new short-term memory.
    pub short_memory_output_value: Neuron<2, F>, // (short_memory, input) -> short_memory_output_value
    pub short_memory_output_gate: Neuron<2, F>   // (long_memory, short_memory_output_value) -> new_short_memory
}

impl<F: Float> LSTMCell<F> {
    /// Amount of parameters stored in the current LSTM cell.
    pub const PARAMS: usize = Neuron::<2, F>::PARAMS * 6;

    /// Create new Long Short-Term Memory cell.
    pub fn random() -> Self {
        Self {
            long_memory_forget_value: Neuron::linear(),
            long_memory_forget_gate: Neuron::linear(),

            long_memory_remember_value: Neuron::linear(),
            long_memory_remember_gate: Neuron::linear(),

            short_memory_output_value: Neuron::linear(),
            short_memory_output_gate: Neuron::linear()
        }
    }

    /// Calculate updated long-term and short-term memory values
    /// (perform forward propagation). Second value is also
    /// the output of the given input.
    pub fn forward(&self, mut long_memory: F, mut short_memory: F, input: F) -> (F, F) {
        // Prepare memory for the neurons' input.
        let mut value_input = Box::new([short_memory, input]);

        // Calculate weighted values for gates.
        let long_memory_forget_value = self.long_memory_forget_value.forward(&value_input);
        let long_memory_remember_value = self.long_memory_remember_value.forward(&value_input);
        let short_memory_output_value = self.short_memory_output_value.forward(&value_input);

        // Stage 1. Long Memory Forget Gate.
        value_input[0] = long_memory;
        value_input[1] = long_memory_forget_value;

        long_memory = self.long_memory_forget_gate.forward(&value_input);

        // Stage 2. Long Memory Remember Gate.
        value_input[0] = long_memory;
        value_input[1] = long_memory_remember_value;

        long_memory = self.long_memory_remember_gate.forward(&value_input);

        // Stage 3. Short Memory Output Gate.
        value_input[0] = long_memory;
        value_input[1] = short_memory_output_value;

        short_memory = self.short_memory_output_gate.forward(value_input);

        // Return updated long-term and short-term memory values.
        (long_memory, short_memory)
    }

    #[inline]
    /// Calculate loss function value.
    pub fn loss(&self, actual_output: F, expected_output: F) -> F {
        self.short_memory_output_gate.loss(actual_output, expected_output)
    }

    /// Update weights of the LSTM cell using given inputs and expected output
    /// (perform backward propagation) and return gradient which could be used
    /// by the previous LSTM cell.
    pub fn backward(
        &mut self,
        long_memory: F,
        short_memory: F,
        input: F,
        output: F,
        policy: &mut BackpropagationSnapshot<'_, { LSTMCell::<F>::PARAMS }, F>
    ) -> F
    where
        [(); LSTMCell::<F>::PARAMS]: Sized,
        [(); Neuron::<2, F>::PARAMS]: Sized
    {
        // --------------------------- Forward pass ---------------------------

        // Prepare memory for the neurons' input.
        let mut value_input = Box::new([short_memory, input]);

        // Calculate weighted values for gates.
        let long_memory_forget_value = self.long_memory_forget_value.forward(&value_input);
        let long_memory_remember_value = self.long_memory_remember_value.forward(&value_input);
        let short_memory_output_value = self.short_memory_output_value.forward(&value_input);

        // Stage 1. Long Memory Forget Gate.
        value_input[0] = long_memory;
        value_input[1] = long_memory_forget_value;

        let forgotten_long_memory = self.long_memory_forget_gate.forward(&value_input);

        // Stage 2. Long Memory Remember Gate.
        value_input[0] = forgotten_long_memory;
        value_input[1] = long_memory_remember_value;

        let remembered_long_memory = self.long_memory_remember_gate.forward(&value_input);

        // Stage 3. Short Memory Output Gate.
        value_input[0] = remembered_long_memory;
        value_input[1] = short_memory_output_value;

        // --------------------------- Backward pass ---------------------------

        // Short Memory Ouput Gate backpropagation.
        let short_memory_output_gate_gradients = policy.window(0, |mut policy| {
            self.short_memory_output_gate.backward(&value_input, output, &mut policy)
        });

        // Short Memory Ouput Value backpropagation.
        value_input[0] = short_memory;
        value_input[1] = input;

        let short_memory_output_value_gradients = policy.window(Neuron::<2, F>::PARAMS, |mut policy| {
            self.short_memory_output_value.backward_propagated(&value_input, short_memory_output_gate_gradients[1], &mut policy)
        });

        // Long Memory Remember Gate backpropagation.
        value_input[0] = forgotten_long_memory;
        value_input[1] = long_memory_remember_value;

        let long_memory_remember_gate_gradients = policy.window(Neuron::<2, F>::PARAMS * 2, |mut policy| {
            self.long_memory_remember_gate.backward_propagated(&value_input, short_memory_output_gate_gradients[0], &mut policy)
        });

        // Long Memory Remember Value backpropagation.
        value_input[0] = short_memory;
        value_input[1] = input;

        let long_memory_remember_value_gradients = policy.window(Neuron::<2, F>::PARAMS * 3, |mut policy| {
            self.long_memory_remember_value.backward_propagated(&value_input, long_memory_remember_gate_gradients[1], &mut policy)
        });

        // Long Memory Forget Gate backpropagation.
        value_input[0] = long_memory;
        value_input[1] = long_memory_forget_value;

        let long_memory_forget_gate_gradients = policy.window(Neuron::<2, F>::PARAMS * 4, |mut policy| {
            self.long_memory_forget_gate.backward_propagated(&value_input, long_memory_remember_gate_gradients[0], &mut policy)
        });

        // Long Memory Forget Value backpropagation.
        value_input[0] = short_memory;
        value_input[1] = input;

        let long_memory_forget_value_gradients = policy.window(Neuron::<2, F>::PARAMS * 5, |mut policy| {
            self.long_memory_forget_value.backward_propagated(&value_input, long_memory_forget_gate_gradients[1], &mut policy)
        });

        // 1. We ignore `long_memory_forget_gate_gradients[0]` because it affects long memory input
        //    which is not something we can change.
        //
        // 2. We calculate mean of both gradients in `short_memory_output_value_gradients`,
        //    `long_memory_remember_value_gradients` and `long_memory_forget_value_gradients`
        //    and return it as the output gradient of this cell.

        let div = F::from_float(1.0 / 6.0_f32);

        // Divide before summing because `F` can be very small type.
        short_memory_output_value_gradients[0]  * div + short_memory_output_value_gradients[1]  * div +
        long_memory_remember_value_gradients[0] * div + long_memory_remember_value_gradients[1] * div +
        long_memory_forget_value_gradients[0]   * div + long_memory_forget_value_gradients[1]   * div
    }

    /// Update weights of the LSTM cell using given inputs and gradient from the
    /// next LSTM cell (perform backward propagation), and return gradient which
    /// could be used by the previous LSTM cell.
    pub fn backward_propagated(
        &mut self,
        long_memory: F,
        short_memory: F,
        input: F,
        output_gradient: F,
        policy: &mut BackpropagationSnapshot<'_, { LSTMCell::<F>::PARAMS }, F>
    ) -> F
    where
        [(); LSTMCell::<F>::PARAMS]: Sized,
        [(); Neuron::<2, F>::PARAMS]: Sized
    {
        // --------------------------- Forward pass ---------------------------

        // Prepare memory for the neurons' input.
        let mut value_input = Box::new([short_memory, input]);

        // Calculate weighted values for gates.
        let long_memory_forget_value = self.long_memory_forget_value.forward(&value_input);
        let long_memory_remember_value = self.long_memory_remember_value.forward(&value_input);
        let short_memory_output_value = self.short_memory_output_value.forward(&value_input);

        // Stage 1. Long Memory Forget Gate.
        value_input[0] = long_memory;
        value_input[1] = long_memory_forget_value;

        let forgotten_long_memory = self.long_memory_forget_gate.forward(&value_input);

        // Stage 2. Long Memory Remember Gate.
        value_input[0] = forgotten_long_memory;
        value_input[1] = long_memory_remember_value;

        let remembered_long_memory = self.long_memory_remember_gate.forward(&value_input);

        // Stage 3. Short Memory Output Gate.
        value_input[0] = remembered_long_memory;
        value_input[1] = short_memory_output_value;

        // --------------------------- Backward pass ---------------------------

        // Short Memory Ouput Gate backpropagation.
        let short_memory_output_gate_gradients = policy.window(0, |mut policy| {
            self.short_memory_output_gate.backward_propagated(&value_input, output_gradient, &mut policy)
        });

        // Short Memory Ouput Value backpropagation.
        value_input[0] = short_memory;
        value_input[1] = input;

        let short_memory_output_value_gradients = policy.window(Neuron::<2, F>::PARAMS, |mut policy| {
            self.short_memory_output_value.backward_propagated(&value_input, short_memory_output_gate_gradients[1], &mut policy)
        });

        // Long Memory Remember Gate backpropagation.
        value_input[0] = forgotten_long_memory;
        value_input[1] = long_memory_remember_value;

        let long_memory_remember_gate_gradients = policy.window(Neuron::<2, F>::PARAMS * 2, |mut policy| {
            self.long_memory_remember_gate.backward_propagated(&value_input, short_memory_output_gate_gradients[0], &mut policy)
        });

        // Long Memory Remember Value backpropagation.
        value_input[0] = short_memory;
        value_input[1] = input;

        let long_memory_remember_value_gradients = policy.window(Neuron::<2, F>::PARAMS * 3, |mut policy| {
            self.long_memory_remember_value.backward_propagated(&value_input, long_memory_remember_gate_gradients[1], &mut policy)
        });

        // Long Memory Forget Gate backpropagation.
        value_input[0] = long_memory;
        value_input[1] = long_memory_forget_value;

        let long_memory_forget_gate_gradients = policy.window(Neuron::<2, F>::PARAMS * 4, |mut policy| {
            self.long_memory_forget_gate.backward_propagated(&value_input, long_memory_remember_gate_gradients[0], &mut policy)
        });

        // Long Memory Forget Value backpropagation.
        value_input[0] = short_memory;
        value_input[1] = input;

        let long_memory_forget_value_gradients = policy.window(Neuron::<2, F>::PARAMS * 5, |mut policy| {
            self.long_memory_forget_value.backward_propagated(&value_input, long_memory_forget_gate_gradients[1], &mut policy)
        });

        // 1. We ignore `long_memory_forget_gate_gradients[0]` because it affects long memory input
        //    which is not something we can change.
        //
        // 2. We calculate mean of both gradients in `short_memory_output_value_gradients`,
        //    `long_memory_remember_value_gradients` and `long_memory_forget_value_gradients`
        //    and return it as the output gradient of this cell.

        let div = F::from_float(1.0 / 6.0_f32);

        // Divide before summing because `F` can be very small type.
        short_memory_output_value_gradients[0]  * div + short_memory_output_value_gradients[1]  * div +
        long_memory_remember_value_gradients[0] * div + long_memory_remember_value_gradients[1] * div +
        long_memory_forget_value_gradients[0]   * div + long_memory_forget_value_gradients[1]   * div
    }

    /// Train LSTM cell on given inputs sequence.
    pub fn train(&mut self, sequence: &[F], policy: &mut BackpropagationSnapshot<'_, { LSTMCell::<F>::PARAMS }, F>)
    where
        [(); LSTMCell::<F>::PARAMS]: Sized,
        [(); Neuron::<2, F>::PARAMS]: Sized
    {
        // Skip training if sequence is too short.
        if sequence.len() < 2 {
            return;
        }

        let n = sequence.len();

        let mut forward = Vec::with_capacity(n);

        forward.push((F::ZERO, F::ZERO));

        // Forward pass.
        for i in 0..n - 2 {
            let (long_memory, short_memory) = forward[i];

            forward.push(self.forward(long_memory, short_memory, sequence[i]));
        }

        // Prepare gradient for backward passes.
        let (long_memory, short_memory) = forward[n - 2];

        let mut gradient = self.backward(long_memory, short_memory, sequence[n - 2], sequence[n - 1], policy);

        // Backward propagation.
        if n > 2 {
            for i in (0..n - 3).rev() {
                let (long_memory, short_memory) = forward[i];

                gradient = self.backward_propagated(long_memory, short_memory, sequence[i], gradient, policy);
            }
        }
    }
}

#[test]
/// Test Long-Short Term Memory Cell training
/// on samples which couldn't be predicted by
/// regular neurons without memory.
fn test_lstm_cell() {
    let mut cell = LSTMCell32::random();

    let mut backpropagation = Backpropagation::default()
        .with_learn_rate(0.003);

    let samples = [
        [1.0, 0.5, 1.0],
        [0.0, 0.5, 0.0],

        [0.9, 0.4, 0.9],
        [0.1, 0.6, 0.1],

        [1.0, 0.4, 1.0],
        [0.0, 0.6, 0.0]
    ];

    for _ in 0..1000 {
        for sequence in samples {
            backpropagation.timestep(|mut policy| {
                cell.train(&sequence, &mut policy);
            });
        }
    }

    let forward_1 = cell.forward(0.0, 0.0, 0.9);
    let forward_2 = cell.forward(0.0, 0.0, 0.1);

    let predicted_1 = cell.forward(forward_1.0, forward_1.1, 0.3);
    let predicted_2 = cell.forward(forward_2.0, forward_2.1, 0.7);

    assert!(predicted_1.1 > 0.8, "expected 0.9");
    assert!(predicted_2.1 < 0.2, "expected 0.1");
}
