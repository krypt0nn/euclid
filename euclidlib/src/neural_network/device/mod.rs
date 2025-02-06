use crate::prelude::*;

pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

pub mod dynamic;

pub mod prelude {
    pub use super::Device;
    pub use super::cpu::CPUDevice;

    #[cfg(feature = "gpu")]
    pub use super::gpu::GPUDevice;

    pub use super::dynamic::DynamicDevice;
}

/// Abstraction over device which can preform calculations.
pub trait Device {
    /// Calculate outputs of given neurons slice (perform forward propagation)
    /// and update outputs vector in place.
    ///
    /// # Safety
    ///
    /// This method uses `alloc` module which depends on system memory allocator.
    /// Application will panic if system memory allocation call will fail.
    unsafe fn forward<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float>(
        &self,
        neurons: &[Neuron<INPUT_SIZE, F>; OUTPUT_SIZE],
        inputs: &[F; INPUT_SIZE],
        outputs: &mut [F; OUTPUT_SIZE]
    ) {
        for i in 0..OUTPUT_SIZE {
            outputs[i] = neurons[i].forward(inputs);
        }
    }

    #[allow(unused_braces)]
    /// Calculate gradients from inputs and expected outputs for given neurons slice
    /// (perform backward propagation) and update them in place.
    ///
    /// # Safety
    ///
    /// This method uses `alloc` module which depends on system memory allocator.
    /// Application will panic if system memory allocation call will fail.
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

        for i in 0..OUTPUT_SIZE {
            let neuron_gradients = policy.window::<{ Neuron::<INPUT_SIZE, F>::PARAMS }, _>({ i * Neuron::<INPUT_SIZE, F>::PARAMS }, |mut policy| {
                neurons[i].backward(&inputs, outputs[i], &mut policy)
            });

            for j in 0..INPUT_SIZE {
                // Divide it here because there's a chance that the generic F type
                // is implemented in a way where we can't accumulate large values
                // so it will overflow at some point and return invalid result.
                gradients[j] += neuron_gradients[j] / div;
            }
        }
    }

    #[allow(unused_braces)]
    /// Calculate backward gradients from inputs and forward gradients
    /// for given neurons slice (perform backward propagation) and update them in place.
    ///
    /// # Safety
    ///
    /// This method uses `alloc` module which depends on system memory allocator.
    /// Application will panic if system memory allocation call will fail.
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

        for i in 0..OUTPUT_SIZE {
            let neuron_gradients = policy.window::<{ Neuron::<INPUT_SIZE, F>::PARAMS }, _>({ i * Neuron::<INPUT_SIZE, F>::PARAMS }, |mut policy| {
                neurons[i].backward_propagated(&inputs, forward_gradients[i], &mut policy)
            });

            for j in 0..INPUT_SIZE {
                // Divide it here because there's a chance that the generic F type
                // is implemented in a way where we can't accumulate large values
                // so it will overflow at some point and return invalid result.
                backward_gradients[j] += neuron_gradients[j] / div;
            }
        }
    }
}
