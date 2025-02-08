pub mod generic_model;
pub mod sized_model;

pub mod prelude {
    pub use super::generic_model::GenericModel as GenericLSTMGenerator;

    pub use super::sized_model::{
        SizedModel as LSTMGenerator,

        TINY_CONTEXT_WINDOW,
        SMALL_CONTEXT_WINDOW,
        MEDIUM_CONTEXT_WINDOW,
        LARGE_CONTEXT_WINDOW,
        HUGE_CONTEXT_WINDOW,
        GIANT_CONTEXT_WINDOW
    };
}
