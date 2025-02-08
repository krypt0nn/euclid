use crate::prelude::*;

/// Layer of `SIZE` Long Short-Term Memory cells.
pub struct LSTMLayer<const SIZE: usize, F: Float> {
    pub cells: Box<[LSTMCell<F>; SIZE]>
}

impl<const SIZE: usize, F: Float> LSTMLayer<SIZE, F> {
    /// Amount of parameters of the current layer.
    pub const PARAMS: usize = LSTMCell::<F>::PARAMS * SIZE;

    /// Create new LSTM cells layer from list of LSTM cells.
    pub fn from_cells(cells: impl IntoHeapArray<LSTMCell<F>, SIZE>) -> Self {
        Self {
            cells: unsafe {
                cells.into_heap_array()
            }
        }
    }

    /// Create new LSTM cells layer with random weights.
    pub fn random() -> Self {
        Self {
            cells: unsafe {
                alloc_fixed_heap_array_from(|_| LSTMCell::random())
                    .expect("Failed to allocate memory for LSTM cells layer")
            }
        }
    }

    /// Calculate new long and short memory values from the given inputs.
    /// Output short-memory values (second array) is also the output
    /// of the layer.
    pub fn forward(
        &self,
        long_memory: impl IntoHeapArray<F, SIZE>,
        short_memory: impl IntoHeapArray<F, SIZE>,
        inputs: impl IntoHeapArray<F, SIZE>
    ) -> (Box<[F; SIZE]>, Box<[F; SIZE]>) {
        unsafe {
            let long_memory = long_memory.into_heap_array();
            let short_memory = short_memory.into_heap_array();
            let inputs = inputs.into_heap_array();

            let mut new_long_memory = alloc_fixed_heap_array()
                .expect("Failed to allocate memory for updated LSTM cells layer long-term memory");

            let mut new_short_memory = alloc_fixed_heap_array()
                .expect("Failed to allocate memory for updated LSTM cells layer short-term memory");

            for i in 0..SIZE {
                let (cell_long_memory, cell_short_memory) = self.cells[i].forward(long_memory[i], short_memory[i], inputs[i]);

                new_long_memory[i] = cell_long_memory;
                new_short_memory[i] = cell_short_memory;
            }

            (new_long_memory, new_short_memory)
        }
    }

    #[allow(unused_braces)]
    /// Update parameters of the LSTM cells layer using provided inputs
    /// and expected outputs, and return list of gradients which could
    /// be used in backward propagation of the previous layer.
    pub fn backward(
        &mut self,
        long_memory: impl IntoHeapArray<F, SIZE>,
        short_memory: impl IntoHeapArray<F, SIZE>,
        inputs: impl IntoHeapArray<F, SIZE>,
        outputs: impl IntoHeapArray<F, SIZE>,
        policy: &mut BackpropagationSnapshot<{ LSTMLayer::<SIZE, F>::PARAMS }, F>
    ) -> Box<[F; SIZE]>
    where
        [(); { LSTMCell::<F>::PARAMS }]: Sized,
        [(); { Neuron::<2, F>::PARAMS }]: Sized
    {
        unsafe {
            let long_memory = long_memory.into_heap_array();
            let short_memory = short_memory.into_heap_array();
            let inputs = inputs.into_heap_array();
            let outputs = outputs.into_heap_array();

            alloc_fixed_heap_array_from(|i| {
                policy.window(LSTMCell::<F>::PARAMS * i, |mut policy| {
                    self.cells[i].backward(long_memory[i], short_memory[i], inputs[i], outputs[i], &mut policy)
                })
            }).expect("Failed to allocate memory for LSTM cells gradients")
        }
    }

    #[allow(unused_braces)]
    /// Update parameters of the LSTM cells layer using provided inputs
    /// and gradients from the following LSTM layer, and return updated list
    /// of gradients which could be used in backward propagation of the previous layer.
    pub fn backward_propagated(
        &mut self,
        long_memory: impl IntoHeapArray<F, SIZE>,
        short_memory: impl IntoHeapArray<F, SIZE>,
        inputs: impl IntoHeapArray<F, SIZE>,
        output_gradients: impl IntoHeapArray<F, SIZE>,
        policy: &mut BackpropagationSnapshot<{ LSTMLayer::<SIZE, F>::PARAMS }, F>
    ) -> Box<[F; SIZE]>
    where
        [(); { LSTMCell::<F>::PARAMS }]: Sized,
        [(); { Neuron::<2, F>::PARAMS }]: Sized
    {
        unsafe {
            let long_memory = long_memory.into_heap_array();
            let short_memory = short_memory.into_heap_array();
            let inputs = inputs.into_heap_array();
            let output_gradients = output_gradients.into_heap_array();

            alloc_fixed_heap_array_from(|i| {
                policy.window(LSTMCell::<F>::PARAMS * i, |mut policy| {
                    self.cells[i].backward_propagated(long_memory[i], short_memory[i], inputs[i], output_gradients[i], &mut policy)
                })
            }).expect("Failed to allocate memory for LSTM cells gradients")
        }
    }

    /// Train LSTM cells layer on given inputs sequence.
    pub fn train(
        &mut self,
        sequence: impl IntoIterator<Item = impl IntoHeapArray<F, SIZE>>,
        policy: &mut BackpropagationSnapshot<'_, { LSTMLayer::<SIZE, F>::PARAMS }, F>
    )
    where
        [(); LSTMCell::<F>::PARAMS]: Sized,
        [(); Neuron::<2, F>::PARAMS]: Sized
    {
        let sequence = sequence.into_iter()
            .map(|sequence| unsafe { sequence.into_heap_array() })
            .collect::<Vec<_>>();

        // Skip training if sequence is too short.
        if sequence.len() < 2 {
            return;
        }

        let n = sequence.len();

        let mut forward = Vec::with_capacity(n);

        let init_memory = unsafe {
            alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for initial LSTM cells memories")
        };

        forward.push((
            init_memory.clone(),
            init_memory
        ));

        // Forward pass.
        for i in 0..n - 2 {
            let (long_memory, short_memory) = &forward[i];

            forward.push(self.forward(long_memory, short_memory, &sequence[i]));
        }

        // Prepare gradient for backward passes.
        let (long_memory, short_memory) = &forward[n - 2];

        let mut gradient = self.backward(long_memory, short_memory, &sequence[n - 2], &sequence[n - 1], policy);

        // Backward propagation.
        if n > 2 {
            for i in (0..n - 3).rev() {
                let (long_memory, short_memory) = &forward[i];

                gradient = self.backward_propagated(long_memory, short_memory, &sequence[i], gradient, policy);
            }
        }
    }
}

#[test]
/// Test Long Short-Term Memory Cells layer.
fn test_lstm_layer() {
    let mut cell = LSTMLayer::random();

    let mut backpropagation = Backpropagation::default()
        .with_learn_rate(0.003);

    let samples = [
        [[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]],
        [[0.0, 1.0], [0.5, 0.5], [0.0, 1.0]],

        [[0.9, 0.1], [0.4, 0.6], [0.9, 0.1]],
        [[0.1, 0.9], [0.6, 0.4], [0.1, 0.9]],

        [[1.0, 0.0], [0.4, 0.6], [1.0, 0.0]],
        [[0.0, 1.0], [0.6, 0.4], [0.0, 1.0]]
    ];

    for _ in 0..1000 {
        for sequence in samples {
            backpropagation.timestep(|mut policy| {
                cell.train(&sequence, &mut policy);
            });
        }
    }

    let forward_1 = cell.forward([0.0, 0.0], [0.0, 0.0], [0.9, 0.1]);
    let forward_2 = cell.forward([0.0, 0.0], [0.0, 0.0], [0.1, 0.9]);

    let predicted_1 = cell.forward(forward_1.0, forward_1.1, [0.3, 0.7]);
    let predicted_2 = cell.forward(forward_2.0, forward_2.1, [0.7, 0.3]);

    assert!(predicted_1.1[0] > 0.8, "expected 0.9");
    assert!(predicted_1.1[1] < 0.2, "expected 0.1");

    assert!(predicted_2.1[0] < 0.2, "expected 0.1");
    assert!(predicted_2.1[1] > 0.8, "expected 0.9");
}
