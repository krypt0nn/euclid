use crate::prelude::*;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CPUDevice;

impl Device for CPUDevice {
    unsafe fn forward<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float>(
        &self,
        neurons: &[Neuron<INPUT_SIZE, F>; OUTPUT_SIZE],
        inputs: &[F; INPUT_SIZE],
        outputs: &mut [F; OUTPUT_SIZE]
    ) {
        let inputs = inputs.into_heap_array();

        std::thread::scope(|scope| {
            let handle = scope.spawn(move || {
                for i in 0..OUTPUT_SIZE {
                    outputs[i] = neurons[i].forward(&inputs);
                }
            });

            handle.join().expect("Failed to join CPU thread");
        });
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

        std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                let div = F::from_float(INPUT_SIZE as f32);

                for i in 0..OUTPUT_SIZE {
                    let neuron_gradients = policy.window::<{ Neuron::<INPUT_SIZE, F>::PARAMS }, _>({ i * Neuron::<INPUT_SIZE, F>::PARAMS }, |mut policy| {
                        neurons[i].backward(&inputs, outputs[i], &mut policy)
                    });

                    for j in 0..INPUT_SIZE {
                        gradients[j] += neuron_gradients[j] / div;
                    }
                }
            });

            handle.join().expect("Failed to join CPU thread");
        });
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

        std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                let div = F::from_float(INPUT_SIZE as f32);

                for i in 0..OUTPUT_SIZE {
                    let neuron_gradients = policy.window::<{ Neuron::<INPUT_SIZE, F>::PARAMS }, _>({ i * Neuron::<INPUT_SIZE, F>::PARAMS }, |mut policy| {
                        neurons[i].backward_propagated(&inputs, forward_gradients[i], &mut policy)
                    });

                    for j in 0..INPUT_SIZE {
                        backward_gradients[j] += neuron_gradients[j] / div;
                    }
                }
            });

            handle.join().expect("Failed to join CPU thread");
        });
    }
}
