use crate::prelude::*;

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
/// This struct has built-in AdamW backpropagation optimization
/// which is enabled by default.
///
/// ```
/// use euclidlib::prelude::*;
///
/// // Create new neuron which is supposed to calculate sum of 2 input numbers.
/// let mut neuron = Neuron32::<2>::linear();
///
/// // Prepare default backpropagation policy.
/// let mut backpropagation = Backpropagation::default()
///     .with_learn_rate(0.03);
///
/// // Train this neuron for 1000 epohs on given examples.
/// for _ in 0..1000 {
///     backpropagation.timestep(|mut policy| {
///         neuron.backward(&[0.0, 1.0], 1.0, &mut policy);
///         neuron.backward(&[2.0, 0.0], 2.0, &mut policy);
///         neuron.backward(&[1.0, 1.0], 2.0, &mut policy);
///         neuron.backward(&[2.0, 1.0], 3.0, &mut policy);
///     });
/// }
///
/// // Validate trained neuron.
/// assert!((neuron.forward(&[3.0, 4.0]) - 7.0).abs() < 0.5);
/// ```
pub struct Neuron<const INPUT_SIZE: usize, F: Float> {
    /// Weights for the neuron inputs.
    pub weights: Box<[F; INPUT_SIZE]>,

    /// Value added to the weighted input sum.
    pub bias: F,

    /// Activation function.
    pub activation_function: fn(F) -> F,

    /// Drivative of the activation function.
    pub activation_function_derivative: fn(F) -> F,

    /// Loss (error) function, in format
    /// actual_output -> expected_output -> value.
    pub loss_function: fn(F, F) -> F,

    /// Derivative of loss function, in format
    /// actual_output -> expected_output -> value.
    pub loss_function_derivative: fn(F, F) -> F
}

impl<const INPUT_SIZE: usize, F: Float> Default for Neuron<INPUT_SIZE, F> {
    fn default() -> Self {
        let weights = unsafe {
            alloc_fixed_heap_array_with(F::EPSILON)
                .expect("Failed to allocate memory for neuron weights")
        };

        Self {
            weights,
            bias: F::EPSILON,

            activation_function: sigmoid,
            activation_function_derivative: sigmoid_derivative,

            loss_function: cross_entropy,
            loss_function_derivative: cross_entropy_derivative
        }
    }
}

impl<const INPUT_SIZE: usize, F: Float> Neuron<INPUT_SIZE, F> {
    /// Amount of parameters stored in the current neuron.
    ///
    /// Obviously equals `INPUT_SIZE + 1` (weights + bias).
    pub const PARAMS: usize = INPUT_SIZE + 1;

    /// Construct new neuron with randomly generated input
    /// weights and bias, and given activation function and
    /// its derivative.
    pub fn random(
        activation_function: fn(F) -> F,
        activation_function_derivative: fn(F) -> F,
        loss_function: fn(F, F) -> F,
        loss_function_derivative: fn(F, F) -> F
    ) -> Self {
        fn xavier_normal(size: f32) -> f32 {
            // Calculate standard deviation for Xavier initialization.
            let std_dev = (2.0 / size).sqrt();

            // Box-Muller transform to convert uniform to normal distribution.
            let u1 = (1.0 - fastrand::f32()).max(f32::EPSILON); // Ensure (0, 1]
            let u2 = fastrand::f32();

            // Generate standard normal variable.
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();

            // Scale to Xavier initialization.
            z0 * std_dev
        }

        let weights = unsafe {
            alloc_fixed_heap_array_from(|_| F::from_float(xavier_normal(INPUT_SIZE as f32 + 2.0)))
                .expect("Failed to allocate memory for neuron weights")
        };

        let bias = F::from_float(xavier_normal(INPUT_SIZE as f32 + 2.0));

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
        Self::random(linear, linear_derivative, quadratic_error, quadratic_error_derivative)
    }

    #[inline]
    /// Call `Neuron::new` with `sigmoid` activation function
    /// and `cross_entropy` loss function.
    pub fn sigmoid() -> Self {
        Self::random(sigmoid, sigmoid_derivative, cross_entropy, cross_entropy_derivative)
    }

    #[inline]
    /// Call `Neuron::new` with `tanh` activation function
    /// and `cross_entropy` loss function.
    pub fn tanh() -> Self {
        Self::random(tanh, tanh_derivative, cross_entropy, cross_entropy_derivative)
    }

    /// Return neuron with `sigmoid` activation function
    /// and `cross_entropy` loss function, and parameters
    /// read from the given bytes slice.
    ///
    /// Use `Neuron::with_activation_function` and
    /// `Neuron::with_loss_function` to change them
    /// to different options.
    pub fn from_bytes(bytes: &[u8; (INPUT_SIZE + 1) * F::BYTES]) -> Self {
        let mut weights = unsafe {
            alloc_fixed_heap_array_with(F::ZERO)
                .expect("Failed to allocate memory for neuron weights")
        };

        let mut bias = [0; F::BYTES];

        bias.copy_from_slice(&bytes[..F::BYTES]);

        let mut i = F::BYTES;
        let mut j = 0;

        while i < (INPUT_SIZE + 1) * F::BYTES {
            let mut weight = [0; F::BYTES];

            weight.copy_from_slice(&bytes[i..i + F::BYTES]);

            weights[j] = F::from_bytes(&weight);

            i += F::BYTES;
            j += 1;
        }

        Self {
            weights,
            bias: F::from_bytes(&bias),

            activation_function: sigmoid,
            activation_function_derivative: sigmoid_derivative,
            loss_function: cross_entropy,
            loss_function_derivative: cross_entropy_derivative
        }
    }

    /// Return bytes slice with current neuron's params.
    ///
    /// Use `Neuron::from_bytes` to restore it.
    pub fn to_bytes(&self) -> Box<[u8; (INPUT_SIZE + 1) * F::BYTES]> {
        let mut bytes = unsafe {
            alloc_fixed_heap_array_with::<u8, { (INPUT_SIZE + 1) * F::BYTES }>(0)
                .expect("Failed to allocate memory for neuron parameters")
        };

        bytes[..F::BYTES].copy_from_slice(&self.bias.to_bytes());

        let mut i = F::BYTES;
        let mut j = 0;

        while i < (INPUT_SIZE + 1) * F::BYTES {
            bytes[i..i + F::BYTES].copy_from_slice(&self.weights[j].to_bytes());

            i += F::BYTES;
            j += 1;
        }

        bytes
    }

    #[inline]
    /// Change weights of the neuron's inputs.
    pub fn with_weights(mut self, weights: [F; INPUT_SIZE]) -> Self {
        self.weights = Box::new(weights);

        self
    }

    #[inline]
    /// Change value added to the weighted sum of the neuron's input.
    pub const fn with_bias(mut self, bias: F) -> Self {
        self.bias = bias;

        self
    }

    #[inline]
    /// Change activation function of the neuron.
    pub const fn with_activation_function(
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
    pub const fn with_loss_function(
        mut self,
        loss_function: fn(F, F) -> F,
        loss_function_derivative: fn(F, F) -> F
    ) -> Self {
        self.loss_function = loss_function;
        self.loss_function_derivative = loss_function_derivative;

        self
    }

    #[inline]
    /// Return params of the current neuron (weights and bias).
    pub const fn params(&self) -> (&[F; INPUT_SIZE], &F) {
        (&self.weights, &self.bias)
    }

    #[inline]
    #[allow(clippy::type_complexity)]
    /// Return activation function and its derivative of the current neuron.
    pub const fn activation_function(&self) -> (fn(F) -> F, fn(F) -> F) {
        (self.activation_function, self.activation_function_derivative)
    }

    #[inline]
    #[allow(clippy::type_complexity)]
    /// Return loss function and its derivative of the current neuron.
    pub const fn loss_function(&self) -> (fn(F, F) -> F, fn(F, F) -> F) {
        (self.loss_function, self.loss_function_derivative)
    }

    /// Resize neuron by truncating input weights or repeating them.
    pub fn resize<const NEW_INPUT_SIZE: usize>(&self) -> Neuron<NEW_INPUT_SIZE, F> {
        if NEW_INPUT_SIZE == INPUT_SIZE {
            return Neuron {
                weights: unsafe {
                    std::mem::transmute::<
                        Box<[F; INPUT_SIZE]>,
                        Box<[F; NEW_INPUT_SIZE]>
                    >(self.weights.clone())
                },

                bias: self.bias,

                activation_function: self.activation_function,
                activation_function_derivative: self.activation_function_derivative,

                loss_function: self.loss_function,
                loss_function_derivative: self.loss_function_derivative
            }
        }

        Neuron {
            weights: unsafe {
                alloc_fixed_heap_array_from(|i| self.weights[i % INPUT_SIZE])
                    .expect("Failed to allocate memory for resized neuron weights")
            },

            bias: self.bias,

            activation_function: self.activation_function,
            activation_function_derivative: self.activation_function_derivative,

            loss_function: self.loss_function,
            loss_function_derivative: self.loss_function_derivative
        }
    }

    /// Convert float type of current neuron to another one (quantize it).
    ///
    /// Note that depending on the actual values of weights and bias
    /// you can lose precision necessary for correct work of the neuron.
    pub fn quantize<T: Float>(&self) -> Neuron<INPUT_SIZE, T> {
        unsafe {
            Neuron {
                weights: Box::new(self.weights.map(T::from_float)),
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
    pub fn make_weighted_input(&self, input: &[F; INPUT_SIZE]) -> F {
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
    pub fn forward(&self, inputs: impl IntoHeapArray<F, INPUT_SIZE>) -> F {
        unsafe {
            (self.activation_function)(self.make_weighted_input(&inputs.into_heap_array()))
        }
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
    ///
    /// Return `Err` on memory allocation failure.
    pub fn backward(
        &mut self,
        inputs: impl IntoHeapArray<F, INPUT_SIZE>,
        output: F,
        policy: &mut BackpropagationSnapshot<'_, { Self::PARAMS }, F>
    ) -> Box<[F; INPUT_SIZE]> {
        unsafe {
            // Allocate all the memory buffers.
            let inputs = inputs.into_heap_array();

            let mut input_values = alloc_fixed_heap_array_with::<F, { Self::PARAMS }>(F::ZERO)
                .expect("Failed to allocate space for neuron backpropagation values");

            let mut forward_gradients = alloc_fixed_heap_array_with::<F, { Self::PARAMS }>(F::ZERO)
                .expect("Failed to allocate space for neuron backpropagation forward gradients");

            let mut backward_gradients = alloc_fixed_heap_array_with::<F, INPUT_SIZE>(F::ZERO)
                .expect("Failed to allocate space for neuron backpropagation backward gradients");

            // Perform backward propagation using low-level API.
            self.backward_in_place(&inputs, output, policy, &mut input_values, &mut forward_gradients, &mut backward_gradients);

            backward_gradients
        }
    }

    #[allow(unused_braces)]
    /// Low-level `Neuron::backward` method variant which doesn't allocate
    /// additional memory and uses only provided containers.
    pub fn backward_in_place(
        &mut self,
        inputs: &[F; INPUT_SIZE],
        output: F,
        policy: &mut BackpropagationSnapshot<'_, { Self::PARAMS }, F>,
        input_values_buf: &mut [F; { Self::PARAMS }],
        forward_gradients_buf: &mut [F; { Self::PARAMS }],
        backward_gradients_buf: &mut [F; INPUT_SIZE]
    ) {
        // Calculate argument of the activation function.
        let argument = self.make_weighted_input(inputs);

        // Calculate activated value returned by the neuron.
        let actual_output = (self.activation_function)(argument);

        // Calculate gradient of the argument.
        let gradient = (self.loss_function_derivative)(actual_output, output) *
            (self.activation_function_derivative)(argument);

        // Set gradients and input values.
        input_values_buf[0] = self.bias;
        forward_gradients_buf[0] = gradient;

        input_values_buf[1..].copy_from_slice(self.weights.as_slice());

        for i in 0..INPUT_SIZE {
            forward_gradients_buf[i + 1] = gradient * inputs[i];
            backward_gradients_buf[i]    = gradient * self.weights[i];
        }

        // Backpropagate weights and bias using calculated gradients.
        policy.backpropagate(input_values_buf, forward_gradients_buf);

        // Update weights and bias from the output values.
        self.bias = input_values_buf[0];
        self.weights.copy_from_slice(&input_values_buf[1..]);
    }

    /// Update weights and bias of the neuron similarly to the
    /// `Neuron::backward` function using values, computed by the
    /// functions called from the following neurons.
    ///
    /// This function will return gradient for the current neuron
    /// which could be used by previous neurons as well.
    ///
    /// It's expected that input slice has `INPUT_SIZE` length.
    /// Returns `INPUT_SIZE` gradients.
    pub fn backward_propagated(
        &mut self,
        inputs: impl IntoHeapArray<F, INPUT_SIZE>,
        output_gradient: F,
        policy: &mut BackpropagationSnapshot<'_, { Self::PARAMS }, F>
    ) -> Box<[F; INPUT_SIZE]> {
        unsafe {
            // Allocate all the memory buffers.
            let inputs = inputs.into_heap_array();

            let mut input_values = alloc_fixed_heap_array_with::<F, { Self::PARAMS }>(F::ZERO)
                .expect("Failed to allocate space for neuron backpropagation values");

            let mut forward_gradients = alloc_fixed_heap_array_with::<F, { Self::PARAMS }>(F::ZERO)
                .expect("Failed to allocate space for neuron backpropagation forward gradients");

            let mut backward_gradients = alloc_fixed_heap_array_with::<F, INPUT_SIZE>(F::ZERO)
                .expect("Failed to allocate space for neuron backpropagation backward gradients");

            // Perform backward propagation using low-level API.
            self.backward_propagated_in_place(&inputs, output_gradient, policy, &mut input_values, &mut forward_gradients, &mut backward_gradients);

            backward_gradients
        }
    }

    #[allow(unused_braces)]
    /// Low-level `Neuron::backward_propagated` method variant which
    /// doesn't allocate additional memory and uses only provided containers.
    pub fn backward_propagated_in_place(
        &mut self,
        inputs: &[F; INPUT_SIZE],
        output_gradient: F,
        policy: &mut BackpropagationSnapshot<'_, { Self::PARAMS }, F>,
        input_values_buf: &mut [F; { Self::PARAMS }],
        forward_gradients_buf: &mut [F; { Self::PARAMS }],
        backward_gradients_buf: &mut [F; INPUT_SIZE]
    ) {
        // Calculate argument of the activation function.
        let argument = self.make_weighted_input(inputs);

        // Calculate gradient of the current neuron using gradient
        // of the connected forward neurons.
        let gradient = output_gradient * (self.activation_function_derivative)(argument);

        // Set gradients and input values.
        input_values_buf[0] = self.bias;
        forward_gradients_buf[0] = gradient;

        input_values_buf[1..].copy_from_slice(self.weights.as_slice());

        for i in 0..INPUT_SIZE {
            forward_gradients_buf[i + 1] = gradient * inputs[i];
            backward_gradients_buf[i]    = gradient * self.weights[i];
        }

        // Backpropagate weights and bias using calculated gradients.
        policy.backpropagate(input_values_buf, forward_gradients_buf);

        // Update weights and bias from the output values.
        self.bias = input_values_buf[0];
        self.weights.copy_from_slice(&input_values_buf[1..]);
    }
}

#[test]
/// Test neuron params conversion from and to bytes slice.
fn test_neuron_bytes() {
    let neuron = Neuron64::<8>::sigmoid();

    let bytes = neuron.to_bytes();

    assert_eq!(bytes.len(), 9 * 8);

    let restored = Neuron64::<8>::from_bytes(&bytes);

    assert_eq!(neuron.params(), restored.params());
}

#[test]
/// Test neuron backward propagation which is supposed
/// to output sum of its inputs.
fn test_neuron_backward_propagation() {
    // Create new neuron which is supposed to calculate sum of 2 input numbers.
    let mut neuron = Neuron32::linear();

    // Prepare backpropagatrion policy for this neuron.
    let mut backpropagation = Backpropagation::default()
        .with_learn_rate(0.003);

    // Train this neuron for 100 epohs on given examples.
    for _ in 0..1000 {
        backpropagation.timestep(|mut policy| {
            neuron.backward(&[0.0, 1.0], 1.0, &mut policy);
            neuron.backward(&[2.0, 0.0], 2.0, &mut policy);
            neuron.backward(&[1.0, 1.0], 2.0, &mut policy);
            neuron.backward(&[2.0, 1.0], 3.0, &mut policy);
        });
    }

    // Validate trained neuron.
    let output = neuron.forward([1.0, 0.5]);

    assert!((output - 1.5).abs() < 0.5);
    assert!(neuron.loss(output, 1.5) < 0.5);

    // Test quantized neuron.
    // qf8_2 will easily store 1.0 which is expected for this neuron.
    let quant_neuron = neuron.quantize::<qf8_4_2>();

    let loss = quant_neuron.diff(&neuron, quadratic_error);

    assert!(loss < 0.05);

    // let output = quant_neuron.forward(&[
    //     qf8_4_2::from_f64(1.0),
    //     qf8_4_2::from_f64(2.0)
    // ]);

    // assert!((output - qf8_4_2::from_f64(1.5)).as_f32().abs() < 1.0);
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

    // Prepare backpropagatrion policies for these neurons.
    let mut policy_11 = Backpropagation::default();
    let mut policy_12 = Backpropagation::default();
    let mut policy_21 = Backpropagation::default();

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

    // Train neurons for 1000 epochs on given examples.
    for _ in 0..1000 {
        for (input_11, input_12, expected_output) in examples {
            // f32 quirks
            let assert_output = ((input_11[0] + input_11[1] + input_12[0] + input_12[1]) * 10.0_f32).round() / 10.0;

            assert_eq!(assert_output, expected_output);

            // Forward pass for the first layer neurons.
            let output_11 = neuron_11.forward(input_11);
            let output_12 = neuron_12.forward(input_12);

            // Backward pass for the output neuron.
            let gradients = policy_21.timestep(|mut policy| {
                neuron_21.backward(&[output_11, output_12], expected_output, &mut policy)
            });

            // Backward pass for the first layer neurons.
            policy_11.timestep(|mut policy| {
                neuron_11.backward_propagated(&input_11, gradients[0], &mut policy);
            });

            policy_12.timestep(|mut policy| {
                neuron_12.backward_propagated(&input_12, gradients[1], &mut policy);
            });
        }
    }

    // Validate trained neurons.
    let a = neuron_11.forward([-1.0, 1.5]);
    let b = neuron_12.forward([0.0, -1.5]);

    let c = neuron_21.forward([a, b]);

    assert!((c + 1.0).abs() < 0.5);
    assert!(neuron_21.loss(c, -1.0) < 0.5);

    // for neuron in [neuron_11, neuron_12, neuron_21] {
    //     assert!((1.0 - neuron.weights[0]) < 0.1);
    //     assert!((1.0 - neuron.weights[1]) < 0.1);
    //     assert!(neuron.bias < 0.1);
    // }
}
