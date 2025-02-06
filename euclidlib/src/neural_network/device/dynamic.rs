use crate::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Dynamic computations device type.
pub enum DynamicDevice {
    CPU(CPUDevice),

    #[cfg(feature = "gpu")]
    GPU(GPUDevice)
}

impl Default for DynamicDevice {
    /// Get best suitable device.
    fn default() -> Self {
        #[cfg(feature = "gpu")]
        if let Some(device) = GPUDevice::new() {
            Self::GPU(device)
        } else {
            Self::CPU(CPUDevice::default())
        }

        #[cfg(not(feature = "gpu"))]
        Self::CPU(CPUDevice::default())
    }
}

impl DynamicDevice {
    #[inline]
    pub fn cpu() -> Self {
        Self::CPU(CPUDevice::default())
    }

    #[cfg(feature = "gpu")]
    #[inline]
    pub fn gpu() -> Option<Self> {
        GPUDevice::new().map(Self::GPU)
    }
}

impl Device for DynamicDevice {
    unsafe fn forward<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float>(
        &self,
        neurons: &[Neuron<INPUT_SIZE, F>; OUTPUT_SIZE],
        inputs: &[F; INPUT_SIZE],
        outputs: &mut [F; OUTPUT_SIZE]
    ) {
        match self {
            Self::CPU(device) => device.forward(neurons, inputs, outputs),

            #[cfg(feature = "gpu")]
            Self::GPU(device) => device.forward(neurons, inputs, outputs)
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
        match self {
            Self::CPU(device) => device.backward(neurons, inputs, outputs, gradients, policy),

            #[cfg(feature = "gpu")]
            Self::GPU(device) => device.backward(neurons, inputs, outputs, gradients, policy)
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
        match self {
            Self::CPU(device) => device.backward_propagated(neurons, inputs, forward_gradients, backward_gradients, policy),

            #[cfg(feature = "gpu")]
            Self::GPU(device) => device.backward_propagated(neurons, inputs, forward_gradients, backward_gradients, policy)
        }
    }
}
