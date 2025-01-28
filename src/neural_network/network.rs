use super::prelude::*;

// struct HiddenNetworkChain<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
//     hidden_layer: Layer<INPUT_SIZE, OUTPUT_SIZE>,
//     follow_up: Option<HiddenNetworkChain<OUTPUT_SIZE, _>>
// }

pub trait NetworkBackend<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> {
    fn forward(&self, input: &[F; INPUT_SIZE]) -> [F; OUTPUT_SIZE];
    fn backward(&mut self, input: &[F; INPUT_SIZE], expected_output: &[F; OUTPUT_SIZE], learn_rate: F);
    fn quantize<T: Float>(&self) -> impl NetworkBackend<INPUT_SIZE, OUTPUT_SIZE, T>;
    fn diff<T: Float>(&self, other: &impl NetworkBackend<INPUT_SIZE, OUTPUT_SIZE, T>, loss_function: fn(f64, f64) -> f64) -> f64;
}

pub struct Network<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> {
    input_layer: Layer<INPUT_SIZE, OUTPUT_SIZE, F>,
    output_layer: Layer<OUTPUT_SIZE, OUTPUT_SIZE, F>
}
