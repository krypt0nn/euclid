use super::prelude::*;

/// Generic `Neuron` type with f32 float type.
pub type Neuron32<const INPUT_SIZE: usize> = Neuron<INPUT_SIZE, f32>;

/// Generic `Neuron` type with f64 float type.
pub type Neuron64<const INPUT_SIZE: usize> = Neuron<INPUT_SIZE, f64>;

#[derive(Debug, Clone, PartialEq)]
/// Single neuron representation.
///
/// Neurons have `INPUT_SIZE` inputs, associated float weights,
/// single float bias and activation function. When input is given
/// neuron sums multiplications of weights and input numbers,
/// adds a bias and uses activation function on the sum to
/// generate single float output. This output is later can be used
/// as one of the inputs of the further neurons. When expected
/// output is given neuron could be trained to return it for
/// provided inputs using technique called backward propagation.
///
/// ```
/// use crate::prelude::*;
///
/// // Create new neuron which is supposed to calculate sum of 2 input numbers.
/// let mut neuron = Neuron32::<2>::linear();
///
/// // Train this neuron for 100 epohs on given examples.
/// for _ in 0..100 {
///     neuron.backward(&[0.0, 1.0], 1.0, 0.15);
///     neuron.backward(&[2.0, 0.0], 2.0, 0.15);
///     neuron.backward(&[1.0, 1.0], 2.0, 0.15);
///     neuron.backward(&[2.0, 1.0], 3.0, 0.15);
/// }
///
/// // Validate trained neuron.
/// assert!((neuron.forward(&[3.0, 4.0]) - 7.0).abs() < 0.5);
/// ```
pub struct Neuron<const INPUT_SIZE: usize, F: Float> {
    /// Weights for the neuron inputs.
    weights: [F; INPUT_SIZE],

    /// Value added to the weighted input sum.
    bias: F,

    /// Activation function.
    activation_function: fn(F) -> F,

    /// Drivative of the activation function.
    activation_function_derivative: fn(F) -> F,

    /// Loss (error) function, in format
    /// actual_output -> expected_output -> value.
    loss_function: fn(F, F) -> F,

    /// Derivative of loss function, in format
    /// actual_output -> expected_output -> value.
    loss_function_derivative: fn(F, F) -> F
}

impl<const INPUT_SIZE: usize, F: Float> Neuron<INPUT_SIZE, F> {
    /// Construct new neuron with randomly generated input
    /// weights and bias, and given activation function and
    /// its derivative.
    pub fn new(
        activation_function: fn(F) -> F,
        activation_function_derivative: fn(F) -> F,
        loss_function: fn(F, F) -> F,
        loss_function_derivative: fn(F, F) -> F
    ) -> Self {
        let mut weights = [F::ZERO; INPUT_SIZE];

        for weight in &mut weights {
            *weight = F::from_float(fastrand::f32() * 2.0 - 1.0);
        }

        let bias = F::from_float(fastrand::f32() * 2.0 - 1.0);

        Self {
            weights,
            bias,
            activation_function,
            activation_function_derivative,
            loss_function,
            loss_function_derivative
        }
    }

    #[inline]
    /// Call `Neuron::new` with `linear` activation function
    /// and `quadratic_error` loss function.
    pub fn linear() -> Self {
        Self::new(linear, linear_derivative, quadratic_error, quadratic_error_derivative)
    }

    #[inline]
    /// Call `Neuron::new` with `sigmoid` activation function
    /// and `cross_entropy` loss function.
    pub fn sigmoid() -> Self {
        Self::new(sigmoid, sigmoid_derivative, cross_entropy, cross_entropy_derivative)
    }

    #[inline]
    /// Call `Neuron::new` with `tanh` activation function
    /// and `cross_entropy` loss function.
    pub fn tanh() -> Self {
        Self::new(tanh, tanh_derivative, cross_entropy, cross_entropy_derivative)
    }

    #[inline]
    /// Change weights of the neuron's inputs.
    pub fn with_weights(mut self, weights: [F; INPUT_SIZE]) -> Self {
        self.weights = weights;

        self
    }

    #[inline]
    /// Change value added to the weighted sum of the neuron's input.
    pub fn with_bias(mut self, bias: F) -> Self {
        self.bias = bias;

        self
    }

    #[inline]
    /// Change activation function of the neuron.
    pub fn with_activation_function(
        mut self,
        activation_function: fn(F) -> F,
        activation_function_derivative: fn(F) -> F
    ) -> Self {
        self.activation_function = activation_function;
        self.activation_function_derivative = activation_function_derivative;

        self
    }

    #[inline]
    /// Change loss function of the neuron.
    pub fn with_loss_function(
        mut self,
        loss_function: fn(F, F) -> F,
        loss_function_derivative: fn(F, F) -> F
    ) -> Self {
        self.loss_function = loss_function;
        self.loss_function_derivative = loss_function_derivative;

        self
    }

    /// Convert float type of current neuron to another one (quantize it).
    ///
    /// Note that depending on the actual values of weights and bias
    /// you can lose precision necessary for correct work of the neuron.
    pub fn quantize<T: Float>(&self) -> Neuron<INPUT_SIZE, T> {
        unsafe {
            Neuron {
                weights: core::array::from_fn::<T, INPUT_SIZE, _>(|i| T::from_float(self.weights[i])),
                bias: T::from_float(self.bias),

                // Transmute activation and loss functions from fn(F) into fn(T).
                // This is safe because these are general Float numbers (trait impls).
                activation_function: std::mem::transmute::<fn(F) -> F, fn(T) -> T>(self.activation_function),
                activation_function_derivative: std::mem::transmute::<fn(F) -> F, fn(T) -> T>(self.activation_function_derivative),
                loss_function: std::mem::transmute::<fn(F, F) -> F, fn(T, T) -> T>(self.loss_function),
                loss_function_derivative: std::mem::transmute::<fn(F, F) -> F, fn(T, T) -> T>(self.loss_function_derivative)
            }
        }
    }

    /// Calculate difference between weights and biases of current
    /// and given neuron using provided loss function.
    ///
    /// This can be used to measure the precision lost by the neuron
    /// after its weights and bias quantization.
    pub fn diff<T: Float>(&self, other: &Neuron<INPUT_SIZE, T>, loss_function: fn(f64, f64) -> f64) -> f64 {
        let weights_diff = self.weights.iter()
            .zip(other.weights.iter())
            .map(|(curr, other)| loss_function(curr.as_f64() - other.as_f64(), 0.0))
            .sum::<f64>();

        weights_diff + loss_function(self.bias.as_f64() - other.bias.as_f64(), 0.0)
    }

    /// Calculate sum of inputs multiplied by appropriate weights
    /// plus the neuron's bias.
    pub fn calc_weighted_input(&self, input: &[F; INPUT_SIZE]) -> F {
        let mut output = F::ZERO;
        let mut i = 0;

        // Sum inputs * weights by 8 at a time.
        // This hints compiler to apply advanced optimizations.
        while i + 8 < INPUT_SIZE {
            output += input[i]     * self.weights[i];
            output += input[i + 1] * self.weights[i + 1];
            output += input[i + 2] * self.weights[i + 2];
            output += input[i + 3] * self.weights[i + 3];
            output += input[i + 4] * self.weights[i + 4];
            output += input[i + 5] * self.weights[i + 5];
            output += input[i + 6] * self.weights[i + 6];
            output += input[i + 7] * self.weights[i + 7];

            i += 8;
        }

        #[allow(clippy::needless_range_loop)]
        // Sum everything remaining.
        while i < INPUT_SIZE {
            output += input[i] * self.weights[i];

            i += 1;
        }

        output + self.bias
    }

    #[inline]
    /// Calculate activated output of the neuron (perform forward propagation).
    pub fn forward(&self, input: &[F; INPUT_SIZE]) -> F {
        (self.activation_function)(self.calc_weighted_input(input))
    }

    #[inline]
    /// Calculate loss function value.
    pub fn loss(&self, actual_output: F, expected_output: F) -> F {
        (self.loss_function)(actual_output, expected_output)
    }

    /// Update weights and bias of the neuron using given inputs,
    /// expected output and learn rate (perform backward propagation).
    ///
    /// This function will return gradients of the propagated input weights
    /// which could be used by previous neurons for backward propagation when
    /// we don't know expected outputs of the neurons.
    pub fn backward(
        &mut self,
        input: &[F; INPUT_SIZE],
        expected_output: F,
        learn_rate: F
    ) -> [F; INPUT_SIZE] {
        // Calculate argument of the activation function.
        let argument = self.calc_weighted_input(input);

        // Calculate activated value returned by the neuron.
        let actual_output = (self.activation_function)(argument);

        // Calculate gradient of the argument.
        let gradient = (self.loss_function_derivative)(actual_output, expected_output) *
            (self.activation_function_derivative)(argument);

        // Prepare map of the backward propagation gradients.
        let mut gradients = [F::ZERO; INPUT_SIZE];

        // Update bias using calculated gradient.
        self.bias -= learn_rate * gradient;

        // Update weights of the neuron's inputs using calculated delta
        // multiplied by the corresponding input, and prepare gradients map which
        // could be used by the previous neurons for backward propagations.
        let mut i = 0;

        // Perform operations by 8 at a time.
        // This hints compiler to apply advanced optimizations.
        while i + 8 < INPUT_SIZE {
            gradients[i]     = gradient * self.weights[i];
            gradients[i + 1] = gradient * self.weights[i + 1];
            gradients[i + 2] = gradient * self.weights[i + 2];
            gradients[i + 3] = gradient * self.weights[i + 3];
            gradients[i + 4] = gradient * self.weights[i + 4];
            gradients[i + 5] = gradient * self.weights[i + 5];
            gradients[i + 6] = gradient * self.weights[i + 6];
            gradients[i + 7] = gradient * self.weights[i + 7];

            self.weights[i]     -= learn_rate * gradient * input[i];
            self.weights[i + 1] -= learn_rate * gradient * input[i + 1];
            self.weights[i + 2] -= learn_rate * gradient * input[i + 2];
            self.weights[i + 3] -= learn_rate * gradient * input[i + 3];
            self.weights[i + 4] -= learn_rate * gradient * input[i + 4];
            self.weights[i + 5] -= learn_rate * gradient * input[i + 5];
            self.weights[i + 6] -= learn_rate * gradient * input[i + 6];
            self.weights[i + 7] -= learn_rate * gradient * input[i + 7];

            i += 8;
        }

        // Do everything else.
        while i < INPUT_SIZE {
            gradients[i] = gradient * self.weights[i];

            self.weights[i] -= learn_rate * gradient * input[i];

            i += 1;
        }

        gradients
    }

    /// Update weights and bias of the neuron similarly to the
    /// `Neuron::backward` function using values, computed by the
    /// functions called from the following neurons.
    ///
    /// This function will return gradient for the current neuron
    /// which could be used by previous neurons as well.
    pub fn backward_propagated(
        &mut self,
        input: &[F; INPUT_SIZE],
        output_gradient: F,
        learn_rate: F
    ) -> [F; INPUT_SIZE] {
        // Calculate argument of the activation function.
        let argument = self.calc_weighted_input(input);

        // Calculate gradient of the current neuron using gradient
        // of the connected forward neurons.
        let gradient = output_gradient * (self.activation_function_derivative)(argument);

        // Prepare map of the backward propagation gradients.
        let mut gradients = [F::ZERO; INPUT_SIZE];

        // Update bias using calculated gradient.
        self.bias -= learn_rate * gradient;

        // Update weights of the neuron's inputs using calculated delta
        // multiplied by the corresponding input, and prepare gradients map which
        // could be used by the previous neurons for backward propagations.
        let mut i = 0;

        // Perform operations by 8 at a time.
        // This hints compiler to apply advanced optimizations.
        while i + 8 < INPUT_SIZE {
            gradients[i]     = gradient * self.weights[i];
            gradients[i + 1] = gradient * self.weights[i + 1];
            gradients[i + 2] = gradient * self.weights[i + 2];
            gradients[i + 3] = gradient * self.weights[i + 3];
            gradients[i + 4] = gradient * self.weights[i + 4];
            gradients[i + 5] = gradient * self.weights[i + 5];
            gradients[i + 6] = gradient * self.weights[i + 6];
            gradients[i + 7] = gradient * self.weights[i + 7];

            self.weights[i]     -= learn_rate * gradient * input[i];
            self.weights[i + 1] -= learn_rate * gradient * input[i + 1];
            self.weights[i + 2] -= learn_rate * gradient * input[i + 2];
            self.weights[i + 3] -= learn_rate * gradient * input[i + 3];
            self.weights[i + 4] -= learn_rate * gradient * input[i + 4];
            self.weights[i + 5] -= learn_rate * gradient * input[i + 5];
            self.weights[i + 6] -= learn_rate * gradient * input[i + 6];
            self.weights[i + 7] -= learn_rate * gradient * input[i + 7];

            i += 8;
        }

        // Do everything else.
        while i < INPUT_SIZE {
            gradients[i] = gradient * self.weights[i];

            self.weights[i] -= learn_rate * gradient * input[i];

            i += 1;
        }

        gradients
    }
}

#[test]
/// Test neuron backward propagation which is supposed
/// to output sum of its inputs.
fn test_neuron_backward_propagation() {
    // Create new neuron which is supposed to calculate sum of 2 input numbers.
    let mut neuron = Neuron32::linear();

    // Train this neuron for 1000 epohs on given examples.
    for _ in 0..1000 {
        neuron.backward(&[0.0, 1.0], 1.0, 0.03);
        neuron.backward(&[2.0, 0.0], 2.0, 0.03);
        neuron.backward(&[1.0, 1.0], 2.0, 0.03);
        neuron.backward(&[2.0, 1.0], 3.0, 0.03);
    }

    // Validate trained neuron.
    let output = neuron.forward(&[3.0, 4.0]);

    assert!((output - 7.0).abs() < 0.5);
    assert!(neuron.loss(output, 7.0) < 0.5);

    assert!((1.0 - neuron.weights[0]) < 0.1);
    assert!((1.0 - neuron.weights[1]) < 0.1);
    assert!(neuron.bias < 0.1);

    // Test quantized neuron.
    // qf8_2 will easily store 1.0 which is expected for this neuron.
    let quant_neuron = neuron.quantize::<qf8_2>();

    let loss = quant_neuron.diff(&neuron, quadratic_error);

    assert!(loss < 0.05);

    let output = quant_neuron.forward(&[
        qf8_2::from_f64(1.0),
        qf8_2::from_f64(2.0)
    ]);

    assert!((output - qf8_2::from_f64(3.0)).as_f32().abs() < 1.0);
}

#[test]
/// Test three linked neurons backward propagation with
/// precalculated gradients which is supposed to output sum of its inputs.
fn test_linked_neurons_backward_propagation() {
    // Create three connected neurons in 2 layers.
    // First layer (neuron_11 and neuron_12) calculate
    // sums of 4 input numbers: first two and second two.
    // Then sums are given to the neuron_12 which sums them
    // as well. The goal is to test backward propagation
    // with gradients calculated in the neuron_12.

    let mut neuron_11 = Neuron32::linear();
    let mut neuron_12 = Neuron32::linear();

    let mut neuron_21 = Neuron32::linear();

    // Examples to learn on (randomly generated).
    let examples = [
        ([-0.4,  0.2], [ 0.1,  0.3],  0.2),
        ([-0.0, -0.3], [ 0.3,  0.3],  0.3),
        ([-0.3,  0.3], [ 0.3,  0.3],  0.6),
        ([-0.1, -0.3], [-0.4,  0.4], -0.4),
        ([-0.3,  0.0], [-0.4,  0.3], -0.4),
        ([-0.2,  0.4], [ 0.2, -0.4],  0.0),
        ([-0.2, -0.1], [ 0.1,  0.0], -0.2),
        ([ 0.1, -0.1], [-0.5, -0.4], -0.9),
        ([-0.3, -0.3], [ 0.4,  0.2], -0.0),
        ([-0.3, -0.3], [-0.4, -0.3], -1.3),
        ([ 0.2, -0.3], [-0.2, -0.1], -0.4),
        ([ 0.1, -0.2], [ 0.5,  0.2],  0.6),
        ([ 0.2, -0.4], [-0.1, -0.2], -0.5),
        ([ 0.1,  0.3], [-0.1, -0.4], -0.1),
        ([-0.5, -0.1], [ 0.1, -0.4], -0.9),
        ([ 0.2,  0.2], [-0.1, -0.2],  0.1),
        ([ 0.1, -0.3], [ 0.4, -0.4], -0.2),
        ([-0.1,  0.0], [ 0.1,  0.2],  0.2),
        ([ 0.1, -0.4], [-0.2,  0.2], -0.3),
        ([ 0.1,  0.4], [-0.2, -0.2],  0.1),
        ([ 0.1,  0.0], [ 0.3, -0.4],  0.0),
        ([ 0.0, -0.2], [ 0.3,  0.3],  0.4),
        ([-0.3, -0.2], [-0.5,  0.1], -0.9),
        ([-0.1, -0.1], [-0.3,  0.4], -0.1),
        ([ 0.3, -0.5], [-0.2,  0.4],  0.0),
        ([-0.2,  0.0], [ 0.3,  0.2],  0.3),
        ([ 0.1,  0.2], [-0.2, -0.2], -0.1),
        ([ 0.1, -0.1], [-0.3,  0.0], -0.3),
        ([ 0.5,  0.1], [-0.4,  0.1],  0.3),
        ([-0.1,  0.3], [ 0.1, -0.4], -0.1),
        ([-0.0, -0.4], [-0.2, -0.4], -1.0),
        ([ 0.4,  0.2], [-0.5,  0.0],  0.1),
        ([-0.0, -0.5], [-0.2, -0.4], -1.1),
        ([ 0.2, -0.3], [ 0.4,  0.0],  0.3),
        ([ 0.3,  0.4], [-0.3,  0.3],  0.7),
        ([ 0.4,  0.0], [-0.1,  0.3],  0.6),
        ([ 0.0, -0.4], [ 0.1,  0.3],  0.0),
        ([-0.1,  0.0], [ 0.1, -0.1], -0.1),
        ([ 0.2,  0.0], [ 0.4, -0.3],  0.3),
        ([-0.3, -0.2], [ 0.4, -0.0], -0.1),
        ([-0.5, -0.4], [-0.2, -0.0], -1.1),
        ([-0.0,  0.3], [-0.3,  0.4],  0.4),
        ([ 0.3,  0.1], [ 0.4, -0.5],  0.3),
        ([ 0.4, -0.2], [-0.5, -0.4], -0.7),
        ([ 0.4,  0.2], [-0.2,  0.3],  0.7),
        ([-0.2,  0.1], [-0.2,  0.1], -0.2),
        ([ 0.2, -0.2], [ 0.2, -0.2],  0.0),
        ([ 0.2, -0.0], [-0.1,  0.0],  0.1),
        ([-0.5, -0.0], [-0.2,  0.2], -0.5),
        ([-0.0, -0.3], [-0.5,  0.4], -0.4)
    ];

    // Train neurons for 100 epochs on given examples.
    for _ in 0..100 {
        for (input_11, input_12, expected_output) in examples {
            // f32 quirks
            let assert_output = ((input_11[0] + input_11[1] + input_12[0] + input_12[1]) * 10.0_f32).round() / 10.0;

            assert_eq!(assert_output, expected_output);

            // Forward pass for the first layer neurons.
            let output_11 = neuron_11.forward(&input_11);
            let output_12 = neuron_12.forward(&input_12);

            // Backward pass for the output neuron.
            let gradients = neuron_21.backward(&[output_11, output_12], expected_output, 0.03);

            // Backward pass for the first layer neurons.
            neuron_11.backward_propagated(&input_11, gradients[0], 0.03);
            neuron_12.backward_propagated(&input_12, gradients[1], 0.03);
        }
    }

    // Validate trained neurons.
    let a = neuron_11.forward(&[-1.0, 1.5]);
    let b = neuron_12.forward(&[0.0, -1.5]);

    let c = neuron_21.forward(&[a, b]);

    assert!((c + 1.0).abs() < 0.5);
    assert!(neuron_21.loss(c, -1.0) < 0.5);

    // for neuron in [neuron_11, neuron_12, neuron_21] {
    //     assert!((1.0 - neuron.weights[0]) < 0.1);
    //     assert!((1.0 - neuron.weights[1]) < 0.1);
    //     assert!(neuron.bias < 0.1);
    // }
}
