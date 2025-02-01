#![feature(generic_const_items, generic_const_exprs)]
#![allow(incomplete_features)]

pub mod alloc;

pub mod document;
pub mod neural_network;
pub mod encoder_decoder;
pub mod tokenizer;

pub mod token;
pub mod message;
pub mod instruction;
pub mod database;

pub mod prelude {
    pub use super::alloc::*;

    pub use super::neural_network::prelude::*;
    pub use super::tokenizer::prelude::*;

    pub use super::encoder_decoder::*;
    pub use super::document::Document;

    pub use super::token::*;
    pub use super::message::*;
    pub use super::instruction::*;
    pub use super::database::prelude::*;
}
