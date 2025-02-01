pub mod documents;
pub mod tokens;

pub mod prelude {
    pub use super::documents::Database as DocumentsDatabase;
    pub use super::tokens::Database as TokensDatabase;
}
