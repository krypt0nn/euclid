use super::prelude::*;

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
/// // Create new layer which is supposed to rotate its inputs.
/// let mut layer = Layer32::<2, 2>::linear();
///
/// // Train this neuron for 100 epohs on given examples.
/// for _ in 0..100 {
///     layer.backward(&[0.0, 1.0], &[1.0, 0.0], 0.03);
///     layer.backward(&[1.0, 0.0], &[0.0, 1.0], 0.03);
///     layer.backward(&[1.0, 1.0], &[1.0, 1.0], 0.03);
///     layer.backward(&[0.5, 0.3], &[0.3, 0.5], 0.03);
/// }
///
/// // Validate trained neuron.
/// let output = layer.forward(&[0.7, 1.0]);
///
/// assert!((output[0] - 1.0).abs() < 0.5);
/// assert!((output[0] - 0.7).abs() < 0.5);
/// ```
pub struct Layer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> {
    neurons: [Neuron<INPUT_SIZE, F>; OUTPUT_SIZE]
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> Layer<INPUT_SIZE, OUTPUT_SIZE, F> {
    #[inline]
    /// Build neurons layer from provided neurons list.
    pub fn from_neurons(neurons: [Neuron<INPUT_SIZE, F>; OUTPUT_SIZE]) -> Self {
        Self {
            neurons
        }
    }

    /// Build neurons layer from randomly generated neurons
    /// with provided activation and loss functions.
    pub fn new(
        activation_function: fn(F) -> F,
        activation_function_derivative: fn(F) -> F,
        loss_function: fn(F, F) -> F,
        loss_function_derivative: fn(F, F) -> F
    ) -> Self {
        Self {
            neurons: core::array::from_fn(|_| {
                Neuron::new(
                    activation_function,
                    activation_function_derivative,
                    loss_function,
                    loss_function_derivative
                )
            })
        }
    }

    #[inline]
    /// Call `Layer::new` with `linear` activation function
    /// and `quadratic_error` loss function.
    pub fn linear() -> Self {
        Self::new(linear, linear_derivative, quadratic_error, quadratic_error_derivative)
    }

    #[inline]
    /// Call `Layer::new` with `sigmoid` activation function
    /// and `cross_entropy` loss function.
    pub fn sigmoid() -> Self {
        Self::new(sigmoid, sigmoid_derivative, cross_entropy, cross_entropy_derivative)
    }

    #[inline]
    /// Call `Layer::new` with `tanh` activation function
    /// and `cross_entropy` loss function.
    pub fn tanh() -> Self {
        Self::new(tanh, tanh_derivative, cross_entropy, cross_entropy_derivative)
    }

    #[inline]
    /// Return neurons of the current layer.
    pub fn neurons(&self) -> &[Neuron<INPUT_SIZE, F>; OUTPUT_SIZE] {
        &self.neurons
    }

    /// Convert float type of all stored neurons to another one (quantize it).
    ///
    /// Note that depending on the actual values of weights and biases
    /// you can lose precision necessary for correct work of the layer.
    pub fn quantize<T: Float>(&self) -> Layer<INPUT_SIZE, OUTPUT_SIZE, T> {
        Layer {
            neurons: core::array::from_fn(|i| {
                self.neurons[i].quantize::<T>()
            })
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

    /// Calculate activated outputs of the neurons (perform forward propagation).
    pub fn forward(&self, input: &[F; INPUT_SIZE]) -> [F; OUTPUT_SIZE] {
        let mut output = [F::ZERO; OUTPUT_SIZE];

        #[allow(clippy::needless_range_loop)]
        for i in 0..OUTPUT_SIZE {
            output[i] = self.neurons[i].forward(input);
        }

        output
    }

    #[inline]
    /// Calculate loss function value.
    pub fn loss(&self, actual_output: &[F; OUTPUT_SIZE], expected_output: &[F; OUTPUT_SIZE]) -> F {
        let mut loss = F::ZERO;

        #[allow(clippy::needless_range_loop)]
        for i in 0..OUTPUT_SIZE {
            loss += self.neurons[i].loss(actual_output[i], expected_output[i]);
        }

        loss
    }

    /// Update weights and biases of the neurons in the current layer
    /// and return their mean gradients for further backward propagation.
    pub fn backward(
        &mut self,
        input: &[F; INPUT_SIZE],
        expected_output: &[F; OUTPUT_SIZE],
        policy: &mut Backpropagation<{ (INPUT_SIZE + 1) * OUTPUT_SIZE }, F>
    ) -> [F; INPUT_SIZE] {
        let mut gradients = [F::ZERO; INPUT_SIZE];

        let div = F::from_float(INPUT_SIZE as f32);

        policy.next_step();

        #[allow(clippy::needless_range_loop)]
        for i in 0..OUTPUT_SIZE {
            let neuron_gradients = policy.window::<{ INPUT_SIZE + 1 }, _>((INPUT_SIZE + 1) * i, |windowed_policy| {
                self.neurons[i].backward(input, expected_output[i], windowed_policy)
            });

            for j in 0..INPUT_SIZE {
                // Divide it here because there's a chance that the generic F type
                // is implemented in a way where we can't accumulate large values
                // so it will overflow at some point and return invalid result.
                gradients[j] += neuron_gradients[j] / div;
            }
        }

        gradients
    }

    /// Update weights and biases of the neurons in the current layer
    /// using gradients provided by the next layer, and return updated
    /// gradients back for the layer staying before the current one.
    pub fn backward_propagated(
        &mut self,
        input: &[F; INPUT_SIZE],
        output_gradient: &[F; OUTPUT_SIZE],
        policy: &mut Backpropagation<{ INPUT_SIZE + 1 }, F>
    ) -> [F; INPUT_SIZE] {
        let mut gradients = [F::ZERO; INPUT_SIZE];

        let div = F::from_float(INPUT_SIZE as f32);

        policy.next_step();

        #[allow(clippy::needless_range_loop)]
        for i in 0..OUTPUT_SIZE {
            let neuron_gradients = policy.window::<{ INPUT_SIZE + 1 }, _>((INPUT_SIZE + 1) * i, |windowed_policy| {
                self.neurons[i].backward_propagated(input, output_gradient[i], windowed_policy)
            });

            for j in 0..INPUT_SIZE {
                // Divide it here because there's a chance that the generic F type
                // is implemented in a way where we can't accumulate large values
                // so it will overflow at some point and return invalid result.
                gradients[j] += neuron_gradients[j] / div;
            }
        }

        gradients
    }
}

#[test]
/// Test simple layer which should return 1.0 and 0.0 if
/// given number is greater than 0.5 and 0.0 and 1.0 otherwise.
fn test_neurons_layer_backward_propagation() {
    // Create simple 2-neuron layer.
    let mut layer = Layer32::<1, 2>::sigmoid();

    // Prepare backpropagatrion policy for this layer.
    let mut policy = Backpropagation::default();

    // Train it on 6 examples for 100 epochs.
    for _ in 0..100 {
        layer.backward(&[1.0], &[1.0, 0.0], &mut policy);
        layer.backward(&[0.0], &[0.0, 1.0], &mut policy);
        layer.backward(&[0.6], &[1.0, 0.0], &mut policy);
        layer.backward(&[0.3], &[0.0, 1.0], &mut policy);
        layer.backward(&[0.8], &[1.0, 0.0], &mut policy);
        layer.backward(&[0.1], &[0.0, 1.0], &mut policy);
    }

    // Validate its output.
    let output = layer.forward(&[0.23]);

    assert!(output[0] < 0.5);
    assert!(output[1] > 0.5);

    assert!(layer.loss(&output, &[0.0, 1.0]) < 0.5);

    // Test quantized neuron.
    let quant_layer = layer.quantize::<qf8_16_1>();

    let loss = quant_layer.diff(&layer, quadratic_error);

    assert!(loss < 0.05);

    let output = quant_layer.forward(&[
        qf8_16_1::from_f64(0.9)
    ]);

    assert!(output[0].as_f32() > 0.5);
    assert!(output[1].as_f32() < 0.5);
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
    let mut policy_input = Backpropagation::default();
    let mut policy_output = Backpropagation::default();

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

    // Train both layers on given samples for 100 epochs.
    for _ in 0..100 {
        for (input, output) in examples {
            let hidden_output = input_layer.forward(&input);

            let gradients = output_layer.backward(&hidden_output, &[output], &mut policy_output);

            input_layer.backward_propagated(&input, &gradients, &mut policy_input);
        }
    }

    // Validate trained layers.
    let hidden_output = input_layer.forward(&[-0.9, -0.3]);

    let output = output_layer.forward(&hidden_output);

    assert!(output[0] > 0.5);
    assert!(output_layer.loss(&output, &[1.0]) < 1.0);
}
