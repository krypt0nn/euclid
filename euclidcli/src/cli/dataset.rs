use std::path::PathBuf;

use clap::Parser;
use colorful::Colorful;

use euclidlib::prelude::*;

#[derive(Parser)]
pub enum DatasetCLI {
    /// Create new dataset.
    Create,

    /// Insert document into the dataset.
    Insert {
        #[arg(long)]
        /// Path to the document file.
        document: PathBuf,

        #[arg(long, default_value_t = String::new())]
        /// Name of the document.
        name: String
    }
}

impl DatasetCLI {
    #[inline]
    pub fn execute(self, path: PathBuf, cache_size: i64) -> anyhow::Result<()> {
        match self {
            Self::Create => {
                let path = path.canonicalize().unwrap_or(path);

                println!("⏳ Creating dataset in {path:?}...");

                match DocumentsDatabase::open(&path, cache_size) {
                    Ok(_) => {
                        println!("{}", "🚀 Dataset created".green());
                        println!("{} {} command will create new database automatically if needed", "📖 Note:".blue(), "`dataset insert`".yellow());
                    }

                    Err(err) => eprintln!("{}", format!("🧯 Failed to create database: {err}").red())
                }
            }

            Self::Insert { document, name } => {
                let database = path.canonicalize().unwrap_or(path);

                println!("⏳ Opening database in {database:?}...");

                match DocumentsDatabase::open(&database, cache_size) {
                    Ok(database) => {
                        let document = document.canonicalize().unwrap_or(document);

                        println!("⏳ Reading document {document:?}...");

                        match std::fs::read_to_string(document) {
                            Ok(document) => {
                                let document = Document::new(document)
                                    .with_name(name);

                                println!("⏳ Inserting document...");

                                match database.insert(&document) {
                                    Ok(_) => println!("{}", "✅ Document inserted".green()),
                                    Err(err) => eprintln!("{}", format!("🧯 Failed to insert document: {err}").red())
                                }
                            }

                            Err(err) => eprintln!("{}", format!("🧯 Failed to read document file: {err}").red())
                        }
                    }

                    Err(err) => eprintln!("{}", format!("🧯 Failed to open database: {err}").red())
                }
            }
        }

        Ok(())
    }
}
