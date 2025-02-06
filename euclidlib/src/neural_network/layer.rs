use crate::prelude::*;

/// Generic `Layer` type with f32 float type.
pub type Layer32<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> = Layer<INPUT_SIZE, OUTPUT_SIZE, f32>;

/// Generic `Layer` type with f64 float type.
pub type Layer64<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> = Layer<INPUT_SIZE, OUTPUT_SIZE, f64>;

#[derive(Debug, Clone, PartialEq)]
/// Group of neurons representation.
///
/// Layers have fixed amount of neurons with same amount
/// of inputs. Layers can connect to each other, when
/// output size of the first layer equals input size
/// of the second layer, and each output of the first layer's
/// neuron is passed to the corresponding input of each neuron
/// of the second layer.
///
/// ```
/// use euclidlib::prelude::*;
///
/// // Create new layer which is supposed to rotate its inputs.
/// let mut layer = Layer32::<2, 2>::linear();
///
/// // Prepare default backpropagation policy.
/// let mut backpropagation = Backpropagation::default()
///     .with_learn_rate(0.03);
///
/// // Train this neuron for 1000 epohs on given examples.
/// for _ in 0..1000 {
///     backpropagation.timestep(|mut policy| {
///         layer.backward(&[0.0, 1.0], &[1.0, 0.0], &mut policy);
///         layer.backward(&[1.0, 0.0], &[0.0, 1.0], &mut policy);
///         layer.backward(&[1.0, 1.0], &[1.0, 1.0], &mut policy);
///         layer.backward(&[0.5, 0.3], &[0.3, 0.5], &mut policy);
///     });
/// }
///
/// // Validate trained neuron.
/// let output = layer.forward(&[0.7, 1.0]);
///
/// assert!((output[0] - 1.0).abs() < 0.5);
/// assert!((output[0] - 0.7).abs() < 0.5);
/// ```
pub struct Layer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> {
    pub neurons: Box<[Neuron<INPUT_SIZE, F>; OUTPUT_SIZE]>
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> Layer<INPUT_SIZE, OUTPUT_SIZE, F> {
    /// Amount of parameters of the current layer.
    pub const PARAMS: usize = Neuron::<INPUT_SIZE, F>::PARAMS * OUTPUT_SIZE;

    #[inline]
    /// Build neurons layer from provided neurons list.
    pub fn from_neurons(neurons: impl IntoHeapArray<Neuron<INPUT_SIZE, F>, OUTPUT_SIZE>) -> Self {
        Self {
            neurons: unsafe {
                neurons.into_heap_array()
            }
        }
    }

    /// Build neurons layer from randomly generated neurons
    /// with provided activation and loss functions.
    pub fn random(
        activation_function: fn(F) -> F,
        activation_function_derivative: fn(F) -> F,
        loss_function: fn(F, F) -> F,
        loss_function_derivative: fn(F, F) -> F
    ) -> Self {
        Self {
            neurons: unsafe {
                alloc_fixed_heap_array_from(|_| {
                    Neuron::random(
                        activation_function,
                        activation_function_derivative,
                        loss_function,
                        loss_function_derivative
                    )
                }).expect("Failed to allocate memory for neurons layer")
            }
        }
    }

    #[inline]
    /// Call `Layer::new` with `linear` activation function
    /// and `quadratic_error` loss function.
    pub fn linear() -> Self {
        Self::random(linear, linear_derivative, quadratic_error, quadratic_error_derivative)
    }

    #[inline]
    /// Call `Layer::new` with `sigmoid` activation function
    /// and `cross_entropy` loss function.
    pub fn sigmoid() -> Self {
        Self::random(sigmoid, sigmoid_derivative, cross_entropy, cross_entropy_derivative)
    }

    #[inline]
    /// Call `Layer::new` with `tanh` activation function
    /// and `cross_entropy` loss function.
    pub fn tanh() -> Self {
        Self::random(tanh, tanh_derivative, cross_entropy, cross_entropy_derivative)
    }

    #[inline]
    /// Return neurons of the current layer.
    pub const fn neurons(&self) -> &[Neuron<INPUT_SIZE, F>; OUTPUT_SIZE] {
        &self.neurons
    }

    /// Resize current layer by truncating neurons and weights or repeating them.
    pub fn resize<const NEW_INPUT_SIZE: usize, const NEW_OUTPUT_SIZE: usize>(&self) -> Layer<NEW_INPUT_SIZE, NEW_OUTPUT_SIZE, F> {
        if INPUT_SIZE == NEW_INPUT_SIZE && OUTPUT_SIZE == NEW_OUTPUT_SIZE {
            return Layer {
                neurons: unsafe {
                    std::mem::transmute::<
                        Box<[Neuron<INPUT_SIZE, F>; OUTPUT_SIZE]>,
                        Box<[Neuron<NEW_INPUT_SIZE, F>; NEW_OUTPUT_SIZE]>
                    >(self.neurons.clone())
                }
            };
        }

        Layer::<NEW_INPUT_SIZE, NEW_OUTPUT_SIZE, F>::from_neurons(unsafe {
            alloc_fixed_heap_array_from(|i| self.neurons[i % OUTPUT_SIZE].resize())
                .expect("Failed to allocate memory for resized neurons layer")
        })
    }

    /// Convert float type of all stored neurons to another one (quantize it).
    ///
    /// Note that depending on the actual values of weights and biases
    /// you can lose precision necessary for correct work of the layer.
    pub fn quantize<T: Float>(&self) -> Layer<INPUT_SIZE, OUTPUT_SIZE, T> {
        Layer {
            neurons: unsafe {
                alloc_fixed_heap_array_from(|i| self.neurons[i].quantize::<T>())
                    .expect("Failed to allocate memory for quantized neurons layer")
            }
        }
    }

    /// Calculate difference between weights and biases of current
    /// and given layers' neurons using provided loss function.
    ///
    /// This can be used to measure the precision lost by the neurons
    /// after their weights and biases quantization.
    pub fn diff<T: Float>(&self, other: &Layer<INPUT_SIZE, OUTPUT_SIZE, T>, loss_function: fn(f64, f64) -> f64) -> f64 {
        self.neurons.iter()
            .zip(other.neurons.iter())
            .map(|(curr, other)| curr.diff(other, loss_function))
            .sum()
    }

    /// Calculate activated outputs of the neurons (perform forward propagation)
    /// using provided calculations device.
    pub fn forward(
        &self,
        inputs: impl IntoHeapArray<F, INPUT_SIZE>,
        device: &impl Device
    ) -> Box<[F; OUTPUT_SIZE]> {
        unsafe {
            let inputs = inputs.into_heap_array();

            let mut outputs = alloc_fixed_heap_array()
                .expect("Failed to allocate memory for neurons layer outputs");

            device.forward(&self.neurons, &inputs, &mut outputs);

            outputs
        }
    }

    #[allow(unused_braces)]
    /// Update weights and biases of the neurons in the current layer
    /// and return their mean gradients for further backward propagation.
    pub fn backward(
        &mut self,
        inputs: impl IntoHeapArray<F, INPUT_SIZE>,
        outputs: impl IntoHeapArray<F, OUTPUT_SIZE>,
        policy: &mut BackpropagationSnapshot<{ Layer::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS }, F>,
        device: &mut impl Device
    ) -> Box<[F; INPUT_SIZE]>
    where
        [(); { Neuron::<INPUT_SIZE, F>::PARAMS }]: Sized
    {
        unsafe {
            let inputs = inputs.into_heap_array();
            let outputs = outputs.into_heap_array();

            let mut gradients = alloc_fixed_heap_array()
                .expect("Failed to allocate memory for neurons gradients");

            device.backward(
                self.neurons.as_mut(),
                &inputs,
                &outputs,
                &mut gradients,
                policy
            );

            gradients
        }
    }

    #[allow(unused_braces)]
    /// Update weights and biases of the neurons in the current layer
    /// using gradients provided by the next layer, and return updated
    /// gradients back for the layer staying before the current one.
    pub fn backward_propagated(
        &mut self,
        inputs: impl IntoHeapArray<F, INPUT_SIZE>,
        forward_gradients: impl IntoHeapArray<F, OUTPUT_SIZE>,
        policy: &mut BackpropagationSnapshot<{ Layer::<INPUT_SIZE, OUTPUT_SIZE, F>::PARAMS }, F>,
        device: &mut impl Device
    ) -> Box<[F; INPUT_SIZE]>
    where
        [(); { Neuron::<INPUT_SIZE, F>::PARAMS }]: Sized
    {
        unsafe {
            let inputs = inputs.into_heap_array();
            let forward_gradients = forward_gradients.into_heap_array();

            let mut backward_gradients = alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for neurons layer backpropagation gradients");

            device.backward_propagated(
                self.neurons.as_mut(),
                &inputs,
                &forward_gradients,
                &mut backward_gradients,
                policy
            );

            backward_gradients
        }
    }

    #[inline]
    /// Calculate loss function value.
    pub fn loss(
        &self,
        actual_output: impl IntoHeapArray<F, OUTPUT_SIZE>,
        expected_output: impl IntoHeapArray<F, OUTPUT_SIZE>
    ) -> F {
        let actual_output = unsafe { actual_output.into_heap_array() };
        let expected_output = unsafe { expected_output.into_heap_array() };

        let mut loss = F::ZERO;

        #[allow(clippy::needless_range_loop)]
        for i in 0..OUTPUT_SIZE {
            loss += self.neurons[i].loss(actual_output[i], expected_output[i]);
        }

        loss
    }
}

#[test]
/// Test simple layer which should return 1.0 and 0.0 if
/// given number is greater than 0.5 and 0.0 and 1.0 otherwise.
fn test_neurons_layer_backward_propagation() {
    // Create simple 2-neuron layer.
    let mut layer = Layer32::<1, 2>::sigmoid();

    // Prepare backpropagatrion policy for this layer.
    let mut backpropagation = Backpropagation::<4, f32>::default()
        .with_learn_rate(0.01);

    // Prepare device.
    let mut device = DynamicDevice::default();

    // Prepare list of train samples.
    let examples = [
        (0.5, [1.0, 0.0]),
        (0.1, [0.0, 1.0]),
        (0.6, [1.0, 0.0]),
        (0.2, [0.0, 1.0]),
        (0.7, [1.0, 0.0]),
        (0.3, [0.0, 1.0]),
        (0.8, [1.0, 0.0]),
        (0.4, [0.0, 1.0])
    ];

    // Train it on given examples for 1000 epochs.
    for _ in 0..1000 {
        for (input, output) in examples {
            backpropagation.timestep(|mut policy| {
                layer.backward(&[input], &output, &mut policy, &mut device);
            });
        }
    }

    // Validate its output.
    let output = layer.forward([0.23], &device);

    assert!(output[0] < 0.5);
    assert!(output[1] > 0.5);

    assert!(layer.loss(&output, [0.0, 1.0]) < 0.5);

    // Test quantized neuron.
    // let quant_layer = layer.quantize::<qf8_4_2>();

    // let loss = quant_layer.diff(&layer, quadratic_error);

    // assert!(loss < 0.05);

    // let output = quant_layer.forward(&[
    //     qf8_4_2::from_f64(0.9)
    // ]);

    // assert!(output[0].as_f32() > 0.5);
    // assert!(output[1].as_f32() < 0.5);
}

#[test]
/// Test simple two-layer network which should return 1.0
/// if two given numbers have the same sign and 0.0 otherwise.
fn test_linked_neurons_layers_backward_propagation() {
    // Create two connected layers - input and output, with 4 neurons
    // in input and 1 neuron in output.
    let mut input_layer = Layer32::<2, 4>::tanh();
    let mut output_layer = Layer32::<4, 1>::sigmoid();

    // Prepare backpropagatrion policy for these layers.
    let mut policy_input = Backpropagation::<{ Layer::<2, 4, f32>::PARAMS }, f32>::default().with_learn_rate(0.001);
    let mut policy_output = Backpropagation::<{ Layer::<4, 1, f32>::PARAMS }, f32>::default().with_learn_rate(0.001);

    // Prepare device.
    let mut device = DynamicDevice::default();

    // Prepare list of train samples.
    let examples = [
        ([ 0.5,  0.3], 1.0),
        ([-0.1,  0.7], 0.0),
        ([-0.3, -0.1], 1.0),
        ([ 0.2, -0.2], 0.0),
        ([ 0.7,  0.5], 1.0),
        ([-0.3,  0.9], 0.0),
        ([-0.5, -0.3], 1.0),
        ([ 0.4, -0.4], 0.0),
        ([ 0.8,  0.6], 1.0),
        ([-0.4,  1.0], 0.0),
        ([-0.9, -0.4], 1.0),
        ([ 0.5, -0.5], 0.0),
        ([ 0.1,  0.2], 1.0),
        ([ 0.2, -0.1], 0.0),
        ([ 0.3,  0.4], 1.0),
        ([-0.2, -0.3], 1.0),
        ([ 0.4, -0.5], 0.0),
        ([ 0.6,  0.7], 1.0),
        ([-0.6,  0.8], 0.0),
        ([ 0.7, -0.8], 0.0),
        ([-0.7, -0.9], 1.0),
        ([ 0.9,  1.0], 1.0),
        ([-1.0,  0.9], 0.0),
        ([ 0.8,  0.7], 1.0),
        ([-0.8, -0.6], 1.0),
        ([ 0.6, -0.4], 0.0),
        ([-0.5,  0.3], 0.0),
        ([ 0.3,  0.2], 1.0),
        ([-0.1,  0.1], 0.0),
        ([ 0.0,  0.0], 1.0)
    ];

    // Train both layers on given samples for 1000 epochs.
    for _ in 0..1000 {
        for (input, output) in examples {
            let hidden_output = input_layer.forward(input, &device);

            let gradients = policy_output.timestep(|mut policy| {
                output_layer.backward(&hidden_output, &[output], &mut policy, &mut device)
            });

            policy_input.timestep(|mut policy| {
                input_layer.backward_propagated(&input, &gradients, &mut policy, &mut device);
            });
        }
    }

    // Validate trained layers.
    let hidden_output = input_layer.forward([-0.9, -0.3], &device);

    let output = output_layer.forward(&hidden_output, &device);

    assert!(output[0] > 0.5);
    assert!(output_layer.loss(&output, [1.0]) < 1.0);
}
