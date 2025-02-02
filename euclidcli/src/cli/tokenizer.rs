use std::io::Write;
use std::rc::Rc;
use std::sync::Mutex;
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
    },

    /// Fit word embeddings model on the provided dataset.
    Train {
        #[arg(long)]
        /// Path to the documents dataset database.
        dataset: PathBuf,

        #[arg(long)]
        /// Convert all the words to lowercase.
        lowercase: bool,

        #[arg(long, default_value_t = 10)]
        /// Amount of backpropagation timesteps to reach the target learn rate value.
        ///
        /// If set to 0 then no warmup will be applied.
        warmup_duration: u32,

        #[arg(long, default_value_t = 0.00005)]
        /// Maximal difference from the target learn rate value.
        ///
        /// Actual learn rate will vary in range `[target - radius, target + radius]`.
        ///
        /// If set to 0 then no cyclic schedule will be applied.
        cycle_radius: f32,

        #[arg(long, default_value_t = 20)]
        /// Amount of timesteps of backpropagation for going full cicle
        /// of learning rate changing.
        ///
        /// If set to 0 then no cyclic schedule will be applied.
        cycle_period: u32,

        #[arg(long, default_value_t = 0.0001)]
        /// Target learn rate of the backpropagation.
        ///
        /// It is different from the actual learn rate because we use cyclic
        /// learn rate update schedule and warmup phase. During warmup learning
        /// rate will slowly increase from `F::EPSILON` to the current value,
        /// and then it will start going lower and higher according to the
        /// cyclic schedule around the target value.
        learn_rate: f32,

        #[arg(long, default_value_t = 0.5)]
        /// Stop training when mean loss fall under this value.
        stop_after_loss: f32,

        #[arg(long, default_value_t = 10)]
        /// Amount of epochs before saving the model and updating token embeddings.
        save_interval: u32
    },

    /// Export tokens and their embeddings to a CSV file.
    Export {
        #[arg(long)]
        /// Path to the csv file.
        csv: PathBuf
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
                                let params = model.params();

                                println!("{}", "✅ Model created".green());
                                println!("              Input tokens: {}", format!("{}", params.input_tokens).yellow());
                                println!("      Embedding dimensions: {}", format!("{}", params.embedding_dimensions).yellow());
                                println!("  Embedding context radius: {}", format!("{}", params.embedding_context_radius).yellow());
                                println!("                Parameters: {}", format!("{}", params.parameters).yellow());

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

                let params = model.params();
                let mut model_size = params.input_tokens as i64;

                println!("{}", "✅ Model loaded".green());
                println!("              Input tokens: {}", format!("{}", params.input_tokens).yellow());
                println!("      Embedding dimensions: {}", format!("{}", params.embedding_dimensions).yellow());
                println!("  Embedding context radius: {}", format!("{}", params.embedding_context_radius).yellow());
                println!("                Parameters: {}", format!("{}", params.parameters).yellow());

                let parser = DocumentsParser::new(lowercase);

                let result = dataset.for_each(move |i, document| {
                    if document.name.is_empty() {
                        println!("⏳ Parsing document №{i}...");
                    } else {
                        println!("⏳ Parsing document №{i} (\"{}\")...", &document.name);
                    }

                    let now = std::time::Instant::now();

                    for token in parser.parse(&document) {
                        let token = database.insert_token(token)?;

                        if token >= model_size {
                            println!("⏳ Tokens limit reached. Upscaling the word embeddings model...");

                            match model.upscale() {
                                Some(scaled) => model = scaled,

                                None => anyhow::bail!("Failed to upscale word embeddings mdoel")
                            }

                            let params = model.params();

                            model_size = params.input_tokens as i64;

                            println!("{}", "✅ Model upscaled".green());
                            println!("              Input tokens: {}", format!("{}", params.input_tokens).yellow());
                            println!("      Embedding dimensions: {}", format!("{}", params.embedding_dimensions).yellow());
                            println!("  Embedding context radius: {}", format!("{}", params.embedding_context_radius).yellow());
                            println!("                Parameters: {}", format!("{}", params.parameters).yellow());

                            println!("⏳ Saving updated model to the database...");

                            database.save_dynamic_model::<f32>(&model)?;

                            println!("{}", "✅ Model saved".green());
                        }

                        let embedding = model.encode(token as usize);

                        database.insert_embedding::<f32>(token, &embedding)?;
                    }

                    println!("{}", format!("✅ Document processed after {:.1} seconds", now.elapsed().as_secs_f32()).green());

                    Ok(())
                });

                if let Err(err) = result {
                    eprintln!("{}", format!("🧯 Failed to process documents from the dataset: {err}").red());

                    return Ok(());
                }
            }

            Self::Train { dataset, lowercase, warmup_duration, cycle_radius, cycle_period, learn_rate, stop_after_loss, save_interval } => {
                let database = path.canonicalize().unwrap_or(path);
                let dataset = dataset.canonicalize().unwrap_or(dataset);

                println!("⏳ Opening tokenizer database in {database:?}...");

                let database = match TokensDatabase::open(&database, cache_size) {
                    Ok(database) => Rc::new(database),
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

                let model = match database.load_dynamic_model::<f32>() {
                    Ok(Some(model)) => Rc::new(Mutex::new(model)),
                    Ok(None) => Rc::new(Mutex::new(WordEmbeddingsModel::<f32>::random_tiny())),

                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to load word embeddings model: {err}").red());

                        return Ok(());
                    }
                };

                let params = model.lock().unwrap().params();

                println!("{}", "✅ Model loaded".green());
                println!("              Input tokens: {}", format!("{}", params.input_tokens).yellow());
                println!("      Embedding dimensions: {}", format!("{}", params.embedding_dimensions).yellow());
                println!("  Embedding context radius: {}", format!("{}", params.embedding_context_radius).yellow());
                println!("                Parameters: {}", format!("{}", params.parameters).yellow());

                let backpropagation = Rc::new(Mutex::new({
                    Backpropagation::<{ GenericWordEmbeddingsModel::<1024, 32, f32>::PARAMS }, f32>::default()
                        .with_warmup_duration(warmup_duration)
                        .with_cycle_radius(cycle_radius)
                        .with_cycle_period(cycle_period)
                        .with_learn_rate(learn_rate)
                }));

                let parser = DocumentsParser::new(lowercase);
                let mut epoch = 1;

                let training_start = std::time::Instant::now();

                loop {
                    println!();
                    println!("📖 Epoch {epoch} ({:.1} minutes)", training_start.elapsed().as_secs_f32() / 60.0);

                    let mean_loss = Rc::new(Mutex::new(0.0));

                    let result = {
                        let database = database.clone();
                        let model = model.clone();
                        let backpropagation = backpropagation.clone();
                        let mean_loss = mean_loss.clone();

                        dataset.for_each(move |i, document| {
                            if document.name.is_empty() {
                                println!("  ⏳ Parsing document №{i}...");
                            } else {
                                println!("  ⏳ Parsing document №{i} (\"{}\")...", &document.name);
                            }

                            let now = std::time::Instant::now();

                            let document = parser.parse(&document)
                                .into_iter()
                                .map(|token| {
                                    match database.query_token(token) {
                                        Ok(Some(token)) => Ok(Some(token as usize)),
                                        Ok(None) => Ok(None),
                                        Err(err) => Err(err)
                                    }
                                })
                                .collect::<Result<Option<Vec<usize>>, _>>()?;

                            let Some(document) = document else {
                                anyhow::bail!("Some tokens of the document are not indexed");
                            };

                            println!("  ⏳ Training on {} parsed tokens...", document.len());

                            backpropagation.lock().unwrap().timestep(|mut policy| {
                                model.lock().unwrap().train::<{ GenericWordEmbeddingsModel::<1024, 32, f32>::PARAMS }>(&document, &mut policy);
                            });

                            let loss = model.lock().unwrap().total_loss(&document);

                            println!("  {}", format!("✅ Done after {:.1} seconds", now.elapsed().as_secs_f32()).green());
                            println!("       Min loss: {}", format!("{:.8}", loss.min_loss).yellow());
                            println!("      Mean loss: {}", format!("{:.8}", loss.mean_loss).yellow());
                            println!("       Max loss: {}", format!("{:.8}", loss.max_loss).yellow());
                            println!("     Total loss: {}", format!("{:.8}", loss.total_loss).yellow());

                            if !loss.total_loss.is_normal() {
                                anyhow::bail!("Calculated loss value is not normal, indicating that the model is broken");
                            }

                            *mean_loss.lock().unwrap() += loss.mean_loss;

                            Ok(())
                        })
                    };

                    let trained_documents = match result {
                        Ok(trained_documents) => trained_documents,
                        Err(err) => {
                            eprintln!("  {}", format!("🧯 Failed to train the model: {err}").red());

                            return Ok(());
                        }
                    };

                    let mean_loss = *mean_loss.lock().unwrap() / trained_documents as f32;

                    if epoch % save_interval == 0 || mean_loss < stop_after_loss {
                        let model = model.lock().unwrap();

                        println!("  ⏳ Saving updated model to the database...");

                        database.save_dynamic_model::<f32>(&model)?;

                        println!("  {}", "✅ Model saved".green());

                        println!("  ⏳ Updating token embeddings...");

                        let tokens = database.for_each(|token, _| {
                            let embedding = model.encode(token as usize);

                            database.insert_embedding::<f32>(token, &embedding)?;

                            Ok(())
                        })?;

                        println!("  {}", format!("✅ Updated {tokens} embeddings").green());
                    }

                    if mean_loss < stop_after_loss {
                        println!();
                        println!("{}", format!("✅ Training completed. Mean loss: {}", format!("{mean_loss:.8}").yellow()).green());

                        return Ok(());
                    }

                    epoch += 1;
                }
            }

            Self::Export { csv } => {
                let database = path.canonicalize().unwrap_or(path);
                let csv = csv.canonicalize().unwrap_or(csv);

                println!("⏳ Opening tokenizer database in {database:?}...");

                let database = match TokensDatabase::open(&database, cache_size) {
                    Ok(database) => Rc::new(database),
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open tokenizer database: {err}").red());

                        return Ok(());
                    }
                };

                let mut file = match std::fs::File::create(&csv) {
                    Ok(file) => file,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to create csv file: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Exporting tokens into {csv:?}...");

                let result = database.clone().for_each(move |token, word| {
                    if let Some(first_char) = word.chars().next() {
                        if first_char.is_alphanumeric() {
                            if let Some(embedding) = database.query_embedding::<f32>(token)? {
                                file.write_all(format!("\"{token}\",\"{word}\"").as_bytes())?;

                                for value in embedding {
                                    file.write_all(format!(",\"{value}\"").as_bytes())?;
                                }

                                file.write_all(b"\n")?;
                            }
                        }
                    }

                    Ok(())
                });

                match result {
                    Ok(tokens) => println!("{}", format!("✅ Exported {tokens} tokens").green()),
                    Err(err) => eprintln!("{}", format!("🧯 Failed to export tokens: {err}").red())
                }
            }
        }

        Ok(())
    }
}
