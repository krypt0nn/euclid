pub mod parser;
pub mod generic_model;
pub mod sized_model;

pub mod prelude {
    pub use super::parser::Parser as DocumentsParser;
    pub use super::generic_model::GenericModel as GenericWordEmbeddingsModel;
    pub use super::sized_model::SizedModel as WordEmbeddingsModel;
}
