pub mod float;
pub mod activations;
pub mod losses;
pub mod backpropagation;
pub mod neuron;
pub mod device;
pub mod layer;

pub mod prelude {
    pub use super::float::*;
    pub use super::activations::*;
    pub use super::losses::*;
    pub use super::backpropagation::*;
    pub use super::neuron::*;
    pub use super::device::prelude::*;
    pub use super::layer::*;
}
