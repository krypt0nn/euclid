pub mod dataset;

use clap::Parser;

#[derive(Parser)]
pub enum Cli {
    /// Manage datasets of plain text corpuses.
    Dataset {
        #[command(subcommand)]
        command: dataset::DatasetCli
    }
}

impl Cli {
    #[inline]
    pub fn execute(self) -> anyhow::Result<()> {
        match self {
            Self::Dataset { command } => command.execute()
        }
    }
}
