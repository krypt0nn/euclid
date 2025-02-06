use std::sync::Arc;

use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Perform computations on CPU using single or multiple threads policy.
///
/// Note that multithreaded policy will allocate additional memory
/// to store computation results within the thread until it's resolved.
pub struct CPUDevice {
    /// Maximal amount of threads `CPUDevice` can allocate for running
    /// a single task. It's recommended to not to make this value too large,
    /// otherwise unreasonable amount of time would be spent on spawning
    /// threads instead of doing usable computations.
    ///
    /// If set to 0 then all the computations will be done in the current thread.
    pub max_threads: usize
}

impl Default for CPUDevice {
    #[inline]
    fn default() -> Self {
        Self {
            max_threads: 0
        }
    }
}

impl Device for CPUDevice {
    unsafe fn forward<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float>(
        &self,
        neurons: &[Neuron<INPUT_SIZE, F>; OUTPUT_SIZE],
        inputs: &[F; INPUT_SIZE],
        outputs: &mut [F; OUTPUT_SIZE]
    ) {
        let inputs = inputs.into_heap_array();

        if self.max_threads == 0 {
            for i in 0..OUTPUT_SIZE {
                outputs[i] = neurons[i].forward(&inputs);
            }
        }

        else {
            std::thread::scope(move |scope| {
                let neurons_per_thread = OUTPUT_SIZE / self.max_threads;
                let neurons_remaining = OUTPUT_SIZE % self.max_threads;

                let inputs = Arc::new(inputs);
                let neurons = Arc::new(neurons);

                let mut handles = Vec::with_capacity(self.max_threads + 1);

                // We expect this thread to resolve faster than other ones.
                let curr_inputs = inputs.clone();
                let curr_neurons = neurons.clone();

                handles.push((OUTPUT_SIZE - neurons_remaining, neurons_remaining, scope.spawn(move || {
                    let mut outputs = Vec::with_capacity(neurons_remaining);

                    for i in OUTPUT_SIZE - neurons_remaining..OUTPUT_SIZE {
                        outputs.push(curr_neurons[i].forward(curr_inputs.as_ref()));
                    }

                    outputs
                })));

                for i in 0..self.max_threads {
                    let curr_inputs = inputs.clone();
                    let curr_neurons = neurons.clone();

                    handles.push((i * neurons_per_thread, neurons_per_thread, scope.spawn(move || {
                        let mut outputs = Vec::with_capacity(neurons_per_thread);

                        for j in i * neurons_per_thread..(i + 1) * neurons_per_thread {
                            outputs.push(curr_neurons[j].forward(curr_inputs.as_ref()));
                        }

                        outputs
                    })));
                }

                for (offset, len, handle) in handles.drain(..) {
                    let handle_outputs = handle.join().expect("Failed to join CPU thread");

                    outputs[offset..offset + len].copy_from_slice(&handle_outputs);
                }
            });
        }
    }

    #[allow(unused_braces)]
    unsafe fn backward<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float>(
        &mut self,
        neurons: &mut [Neuron<INPUT_SIZE, F>; OUTPUT_SIZE],
        inputs: &[F; INPUT_SIZE],
        outputs: &[F; OUTPUT_SIZE],
        gradients: &mut [F; INPUT_SIZE],
        policy: &mut BackpropagationSnapshot<{ Layer::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS }, F>
    ) where [(); { Neuron::<INPUT_SIZE, F>::PARAMS }]: Sized {
        let inputs = inputs.into_heap_array();
        let outputs = outputs.into_heap_array();

        let div = F::from_float(INPUT_SIZE as f32);

        // TODO: multithread code

        for i in 0..OUTPUT_SIZE {
            let neuron_gradients = policy.window::<{ Neuron::<INPUT_SIZE, F>::PARAMS }, _>({ i * Neuron::<INPUT_SIZE, F>::PARAMS }, |mut policy| {
                neurons[i].backward(&inputs, outputs[i], &mut policy)
            });

            for j in 0..INPUT_SIZE {
                gradients[j] += neuron_gradients[j] / div;
            }
        }
    }

    #[allow(unused_braces)]
    unsafe fn backward_propagated<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float>(
        &mut self,
        neurons: &mut [Neuron<INPUT_SIZE, F>; OUTPUT_SIZE],
        inputs: &[F; INPUT_SIZE],
        forward_gradients: &[F; OUTPUT_SIZE],
        backward_gradients: &mut [F; INPUT_SIZE],
        policy: &mut BackpropagationSnapshot<{ Layer::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS }, F>
    ) where [(); { Neuron::<INPUT_SIZE, F>::PARAMS }]: Sized {
        let inputs = inputs.into_heap_array();
        let forward_gradients = forward_gradients.into_heap_array();

        let div = F::from_float(INPUT_SIZE as f32);

        // TODO: multithread code

        for i in 0..OUTPUT_SIZE {
            let neuron_gradients = policy.window::<{ Neuron::<INPUT_SIZE, F>::PARAMS }, _>({ i * Neuron::<INPUT_SIZE, F>::PARAMS }, |mut policy| {
                neurons[i].backward_propagated(&inputs, forward_gradients[i], &mut policy)
            });

            for j in 0..INPUT_SIZE {
                backward_gradients[j] += neuron_gradients[j] / div;
            }
        }
    }
}
