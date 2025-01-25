use super::prelude::*;

// struct HiddenNetworkChain<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
//     hidden_layer: Layer<INPUT_SIZE, OUTPUT_SIZE>,
//     follow_up: Option<HiddenNetworkChain<OUTPUT_SIZE, _>>
// }

pub struct Network<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, F: Float> {
    input_layer: Layer<INPUT_SIZE, OUTPUT_SIZE, F>,
    output_layer: Layer<OUTPUT_SIZE, OUTPUT_SIZE, F>
}
