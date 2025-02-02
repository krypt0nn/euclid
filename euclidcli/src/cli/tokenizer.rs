use std::path::PathBuf;

use clap::Parser;
use colorful::Colorful;

use euclidlib::prelude::*;

#[derive(Parser)]
pub enum TokenizerCLI {
    /// Create new tokenizer model.
    Create,

    /// Update tokens list and their embeddings in the tokenizer model.
    Update {
        #[arg(long)]
        /// Path to the documents dataset database.
        dataset: PathBuf,

        #[arg(long)]
        /// Convert all the words to lowercase.
        lowercase: bool
    }
}

impl TokenizerCLI {
    #[inline]
    pub fn execute(self, path: PathBuf, cache_size: i64) -> anyhow::Result<()> {
        match self {
            Self::Create => {
                let path = path.canonicalize().unwrap_or(path);

                println!("⏳ Creating database in {path:?}...");

                match TokensDatabase::open(&path, cache_size) {
                    Ok(db) => {
                        match db.saved_model_params() {
                            Ok(Some(_)) => println!("{} Database already existed", "📖 Note:".blue()),

                            Ok(None) => {
                                println!("⏳ Creating tiny word embeddings model...");

                                let model = WordEmbeddingsModel::<f32>::random_tiny();

                                let (inputs, embeddings, params) = model.params();

                                println!("✅ Model created: {inputs} inputs, {embeddings} embedding dimensions, {params} parameters");
                                println!("⏳ Saving the model to the database...");

                                match db.save_dynamic_model::<f32>(&model) {
                                    Ok(()) => println!("{}", "🚀 Database created".green()),
                                    Err(err) => eprintln!("{}", format!("🧯 Failed to save the model: {err}").red())
                                }
                            },

                            Err(err) => eprintln!("{}", format!("🧯 Failed to read created database: {err}").red())
                        }
                    }

                    Err(err) => eprintln!("{}", format!("🧯 Failed to create database: {err}").red())
                }
            }

            Self::Update { dataset, lowercase } => {
                let database = path.canonicalize().unwrap_or(path);
                let dataset = dataset.canonicalize().unwrap_or(dataset);

                println!("⏳ Opening tokenizer database in {database:?}...");

                let database = match TokensDatabase::open(&database, cache_size) {
                    Ok(database) => database,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open tokenizer database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening documents dataset in {dataset:?}...");

                let dataset = match DocumentsDatabase::open(&dataset, cache_size) {
                    Ok(dataset) => dataset,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open documents dataset: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Loading word embeddings model...");

                let mut model = match database.load_dynamic_model::<f32>() {
                    Ok(Some(model)) => model,

                    Ok(None) => WordEmbeddingsModel::<f32>::random_tiny(),

                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to load word embeddings model: {err}").red());

                        return Ok(());
                    }
                };

                let mut model_size = model.params().0 as i64;

                let parser = DocumentsParser::new(lowercase);

                let result = dataset.for_each(move |i, document| {
                    println!("⏳ Processing document №{i}...");

                    let now = std::time::Instant::now();

                    for token in parser.parse(&document) {
                        let token = database.insert_token(token)?;

                        if token >= model_size {
                            println!("⏳ Tokens limit reached. Upscaling the word embeddings model...");

                            match model.upscale() {
                                Some(scaled) => model = scaled,

                                None => anyhow::bail!("Failed to upscale word embeddings mdoel")
                            }

                            let (new_inputs, new_embeddings, new_params) = model.params();

                            model_size = new_inputs as i64;

                            println!("✅ Model upscaled: {new_inputs} inputs, {new_embeddings} embedding dimensions, {new_params} parameters");
                            println!("⏳ Saving updated model to the database...");

                            database.save_dynamic_model::<f32>(&model)?;

                            println!("{}", "✅ Model saved".green());
                        }

                        let embedding = model.encode(token as usize);

                        database.insert_embedding::<f32>(token, &embedding)?;
                    }

                    println!("{}", format!("✅ Document processed after {:.1} seconds", now.elapsed().as_millis() as f64 / 1000.0).green());

                    Ok(())
                });

                if let Err(err) = result {
                    eprintln!("{}", format!("🧯 Failed to process documents from the dataset: {err}").red());

                    return Ok(());
                }
            }
        }

        Ok(())
    }
}
