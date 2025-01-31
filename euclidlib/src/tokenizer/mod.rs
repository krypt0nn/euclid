pub mod parser;
pub mod model;

pub mod prelude {
    pub use super::parser::Parser as DocumentsParser;
    pub use super::model::Model as WordEmbeddingsModel;
}
