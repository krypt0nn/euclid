use std::path::PathBuf;

use clap::Parser;
use colorful::Colorful;

use euclidlib::prelude::*;

#[derive(Parser)]
pub enum DatasetCli {
    /// Create new dataset.
    Create {
        #[arg(long, short)]
        /// Path to the database file.
        path: PathBuf,

        #[arg(long, default_value_t = -4096)]
        /// SQLite database cache size.
        ///
        /// Positive value sets cache size in bytes, negative - in sqlite pages.
        cache_size: i64
    },

    /// Insert document into the dataset.
    Insert {
        #[arg(long)]
        /// Path to the database file.
        database: PathBuf,

        #[arg(long)]
        /// Path to the document file.
        document: PathBuf,

        #[arg(long, default_value_t = String::new())]
        /// Name of the document.
        name: String,

        #[arg(long, default_value_t = -4096)]
        /// SQLite database cache size.
        ///
        /// Positive value sets cache size in bytes, negative - in sqlite pages.
        cache_size: i64
    }
}

impl DatasetCli {
    #[inline]
    pub fn execute(self) -> anyhow::Result<()> {
        match self {
            Self::Create { path, cache_size } => {
                match DocumentsDatabase::open(&path, cache_size) {
                    Ok(_) => {
                        let path = path.canonicalize().unwrap_or(path);

                        println!("{}", format!("🚀 Database created in {path:?}").green());
                        println!("{} {} command will create new database automatically if needed", "📖 Note:".blue(), "`dataset insert`".yellow());
                    }

                    Err(err) => eprintln!("{}", format!("🧯 Failed to create database: {err}").red())
                }
            }

            Self::Insert { database, document, name, cache_size } => {
                let database = database.canonicalize().unwrap_or(database);

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
