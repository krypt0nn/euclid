use super::prelude::*;

macro_rules! network {
    // Entry point: start with empty layers and count=1
    ($name:ident $($rest:tt)*) => {
        network!(@parse $name [] 1 $($rest)*);
    };

    // Parse layer config with -> separator
    (@parse $name:ident [$($layers:tt)*] $count:tt [ $in:expr, $act:expr, $act_deriv:expr, $loss:expr, $loss_deriv:expr ] -> $($rest:tt)*) => {
        network!(@process_next $name [$($layers)* ($count, $in, $act, $act_deriv, $loss, $loss_deriv)] $count $($rest)*);
    };

    // Process next token after ->
    (@process_next $name:ident [$($layers:tt)*] $count:tt [ $out:expr ]) => {
        network!(@generate $name [$($layers)*] $count $out);
    };
    (@process_next $name:ident [$($layers:tt)*] $count:tt [ $out:expr, $next_act:expr, $next_act_deriv:expr, $next_loss:expr, $next_loss_deriv:expr ] -> $($rest:tt)*) => {
        network!(@parse $name [$($layers)*] ($count + 1) [ $out, $next_act, $next_act_deriv, $next_loss, $next_loss_deriv ] -> $($rest)*);
    };

    // Generate struct and impl
    (@generate $name:ident [ $( ($layer_num:tt, $in:expr, $act:expr, $act_deriv:expr, $loss:expr, $loss_deriv:expr) ),* ] $last_count:tt $out:expr) => {
        pub struct $name<F: Float> {
            $(
                pub layer_$layer_num: Layer<$in, $out, F>,
            )*
        }

        impl<F: Float> Default for $name<F> {
            fn default() -> Self {
                Self {
                    $(
                        layer_$layer_num: Layer::new(
                            $act,
                            $act_deriv,
                            $loss,
                            $loss_deriv
                        ),
                    )*
                }
            }
        }
    };
}

network!(Network1 [4, tanh, tanh_derivative, cross_entropy, cross_entropy_derivative] -> 2);
network!(Network2 [4, tanh, tanh_derivative, cross_entropy, cross_entropy_derivative] -> [6, sigmoid, sigmoid_derivative, cross_entropy, cross_entropy_derivative] -> 2);

// network!(Network2 [4, tanh, tanh_derivative, cross_entropy, cross_entropy_derivative] -> [6, sigmoid, sigmoid_derivative, cross_entropy, cross_entropy_derivative] -> 2);

// // struct HiddenNetworkChain<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
// //     hidden_layer: Layer<INPUT_SIZE, OUTPUT_SIZE>,
// //     follow_up: Option<HiddenNetworkChain<OUTPUT_SIZE, _>>
// // }

// /// Backend trait for statically allocated neural networks.
// ///
// /// Implements very basic logic which could be used by the `Network`
// /// struct for more advanced operations. Needed to allow `Network`
// /// to be constant size with statically allocatable memory while
// /// being able to have dynamic number of layers and their sizes.
// pub trait StaticNetworkBackend<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> {
//     /// Calculate output values from the given input (perform forward propagation).
//     fn forward(&self, input: &[F; INPUT_SIZE]) -> [F; OUTPUT_SIZE];

//     /// Update weights and biases of the neurons in the current network
//     /// and return their mean gradients for further backward propagation.
//     fn backward<const BACKPROPAGATION_SIZE: usize>(
//         &mut self,
//         input: &[F; INPUT_SIZE],
//         expected_output: &[F; OUTPUT_SIZE],
//         policy: &mut Backpropagation<BACKPROPAGATION_SIZE, F>
//     ) -> [F; INPUT_SIZE];

//     /// Update weights and biases of the neurons in the current network
//     /// using gradients provided by the next layer, and return updated
//     /// gradients back for the layer staying before the current one.
//     fn backward_propagated<const BACKPROPAGATION_SIZE: usize>(
//         &mut self,
//         input: &[F; INPUT_SIZE],
//         output_gradient: &[F; OUTPUT_SIZE],
//         policy: &mut Backpropagation<BACKPROPAGATION_SIZE, F>
//     ) -> [F; INPUT_SIZE];
// }

// /// Final layer generic for statically allocatable neural network backend.
// pub struct NetworkFinalLayerBackend<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> {
//     layer: Layer<INPUT_SIZE, OUTPUT_SIZE, F>
// }

// impl<
//     const INPUT_SIZE: usize,
//     const OUTPUT_SIZE: usize,
//     F: Float
// > StaticNetworkBackend<INPUT_SIZE, OUTPUT_SIZE, F> for NetworkFinalLayerBackend<
//     INPUT_SIZE,
//     OUTPUT_SIZE,
//     F
// > {
//     #[inline]
//     fn forward(&self, input: &[F; INPUT_SIZE]) -> [F; OUTPUT_SIZE] {
//         self.layer.forward(input)
//     }

//     #[inline]
//     fn backward<const BACKPROPAGATION_SIZE: usize>(
//         &mut self,
//         input: &[F; INPUT_SIZE],
//         expected_output: &[F; OUTPUT_SIZE],
//         policy: &mut Backpropagation<BACKPROPAGATION_SIZE, F>
//     ) -> [F; INPUT_SIZE] {
//         policy.window::<{ (INPUT_SIZE + 1) * OUTPUT_SIZE }, _>(0, |policy: &mut Backpropagation<{ (INPUT_SIZE + 1) * OUTPUT_SIZE }, _>| {
//             self.layer.backward(input, expected_output, policy)
//         })
//     }

//     #[inline]
//     fn backward_propagated<const BACKPROPAGATION_SIZE: usize>(
//         &mut self,
//         input: &[F; INPUT_SIZE],
//         output_gradient: &[F; OUTPUT_SIZE],
//         policy: &mut Backpropagation<BACKPROPAGATION_SIZE, F>
//     ) -> [F; INPUT_SIZE] {
//         self.layer.backward_propagated(input, output_gradient, policy)
//     }
// }

// /// Intermediate layer generic for statically allocatable neural network backend.
// pub struct NetworkChainedLayerBackend<
//     const INPUT_SIZE: usize,
//     const HIDDEN_SIZE: usize,
//     const OUTPUT_SIZE: usize,
//     F: Float,
//     Chained: StaticNetworkBackend<HIDDEN_SIZE, OUTPUT_SIZE, F>
// > {
//     input_layer: Layer<INPUT_SIZE, HIDDEN_SIZE, F>,
//     chained_network: Chained
// }

// impl<
//     const INPUT_SIZE: usize,
//     const HIDDEN_SIZE: usize,
//     const OUTPUT_SIZE: usize,
//     F: Float,
//     Chained: StaticNetworkBackend<HIDDEN_SIZE, OUTPUT_SIZE, F>
// > StaticNetworkBackend<INPUT_SIZE, OUTPUT_SIZE, F> for NetworkChainedLayerBackend<
//     INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
//     F, Chained
// > {
//     fn forward(&self, input: &[F; INPUT_SIZE]) -> [F; OUTPUT_SIZE] {
//         self.chained_network.forward(&self.input_layer.forward(input))
//     }

//     fn backward(
//         &mut self,
//         input: &[F; INPUT_SIZE],
//         expected_output: &[F; OUTPUT_SIZE],
//         policy: &mut Backpropagation<{ (INPUT_SIZE + 1) * HIDDEN_SIZE + (HIDDEN_SIZE + 1) * OUTPUT_SIZE }, F>
//     ) -> [F; INPUT_SIZE] {
//         let chained_input = self.input_layer.forward(input);
//         let gradients = self.chained_network.backward(&chained_input, expected_output, policy);

//         self.input_layer.backward_propagated(input, &gradients, &policy)
//     }

//     fn backward_propagated(
//         &mut self,
//         input: &[F; INPUT_SIZE],
//         output_gradient: &[F; OUTPUT_SIZE],
//         policy: &mut Backpropagation<{ INPUT_SIZE + 1 }, F>
//     ) -> [F; INPUT_SIZE] {
//         todo!()
//     }
// }

// /// Static neural network backend with input, output and 1 hidden layer of given sizes.
// pub type H2Backend<const INPUT_SIZE: usize, const H1: usize, const OUTPUT_SIZE: usize, F> = NetworkChainedLayerBackend<
//     INPUT_SIZE, H1, OUTPUT_SIZE, F,
//     NetworkFinalLayerBackend<H1, OUTPUT_SIZE, F>
// >;

// // type H3Backend<const INPUT_SIZE: usize, const H1: usize, const H2: usize, const OUTPUT_SIZE: usize, F: Float> = NetworkChainedLayerBackend<
// //     INPUT_SIZE, H1, OUTPUT_SIZE

// // >

// pub struct Network<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float, Backend: StaticNetworkBackend<INPUT_SIZE, OUTPUT_SIZE, F>> {
//     _float: std::marker::PhantomData<F>,
//     backend: Backend
// }

// impl<
//     const INPUT_SIZE: usize,
//     const OUTPUT_SIZE: usize,
//     F: Float,
//     Backend: StaticNetworkBackend<INPUT_SIZE, OUTPUT_SIZE, F>
// > StaticNetworkBackend<INPUT_SIZE, OUTPUT_SIZE, F> for Network<INPUT_SIZE, OUTPUT_SIZE, F, Backend> {
//     #[inline]
//     fn forward(&self, input: &[F; INPUT_SIZE]) -> [F; OUTPUT_SIZE] {
//         self.backend.forward(input)
//     }

//     #[inline]
//     fn backward(
//         &mut self,
//         input: &[F; INPUT_SIZE],
//         expected_output: &[F; OUTPUT_SIZE],
//         policy: &mut Backpropagation<{ (INPUT_SIZE + 1) * OUTPUT_SIZE }, F>
//     ) -> [F; INPUT_SIZE] {
//         self.backend.backward(input, expected_output, policy)
//     }

//     #[inline]
//     fn backward_propagated(
//         &mut self,
//         input: &[F; INPUT_SIZE],
//         output_gradient: &[F; OUTPUT_SIZE],
//         policy: &mut Backpropagation<{ INPUT_SIZE + 1 }, F>
//     ) -> [F; INPUT_SIZE] {
//         self.backend.backward_propagated(input, output_gradient, policy)
//     }
// }
