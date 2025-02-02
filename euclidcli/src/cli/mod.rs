use std::path::PathBuf;

use clap::Parser;

pub mod dataset;
pub mod tokenizer;

#[derive(Parser)]
pub enum CLI {
    /// Manage datasets of plain text corpuses.
    Dataset {
        #[arg(long, short)]
        /// Path to the database file.
        path: PathBuf,

        #[arg(long, default_value_t = -4096)]
        /// SQLite database cache size.
        ///
        /// Positive value sets cache size in bytes, negative - in sqlite pages.
        cache_size: i64,

        #[command(subcommand)]
        command: dataset::DatasetCLI
    },

    /// Manage text tokenizing models.
    Tokenizer {
        #[arg(long, short)]
        /// Path to the database file.
        path: PathBuf,

        #[arg(long, default_value_t = -4096)]
        /// SQLite database cache size.
        ///
        /// Positive value sets cache size in bytes, negative - in sqlite pages.
        cache_size: i64,

        #[command(subcommand)]
        command: tokenizer::TokenizerCLI
    },
}

impl CLI {
    #[inline]
    pub fn execute(self) -> anyhow::Result<()> {
        match self {
            Self::Dataset { command, path, cache_size } => command.execute(path, cache_size),
            Self::Tokenizer { command, path, cache_size } => command.execute(path, cache_size)
        }
    }
}
