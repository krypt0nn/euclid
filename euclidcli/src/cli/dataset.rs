use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
pub enum DatasetCli {
    /// Create new dataset.
    Create {
        /// Path to the dataset file.
        path: PathBuf
    }
}

impl DatasetCli {
    #[inline]
    pub fn execute(self) -> anyhow::Result<()> {
        match self {
            Self::Create { path } => {
                todo!()
            }
        }
    }
}
