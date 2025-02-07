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
        name: String,

        #[arg(long)]
        /// Convert content of the document to lowercase.
        lowercase: bool,

        #[arg(long)]
        /// Read input file as discord chat history export in JSON format.
        discord_chat: bool,

        #[arg(long)]
        /// Split discord chat messages into separate documents.
        discord_split_documents: bool,

        #[arg(long, default_value_t = 0)]
        /// Use last N messages from the discord chat history.
        ///
        /// When 0 is set (default), then all messages are used.
        discord_last_n: usize
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

            Self::Insert { document, name, lowercase, discord_chat, discord_split_documents, discord_last_n } => {
                let database = path.canonicalize().unwrap_or(path);

                println!("⏳ Opening database in {database:?}...");

                match DocumentsDatabase::open(&database, cache_size) {
                    Ok(database) => {
                        let document = document.canonicalize().unwrap_or(document);

                        println!("⏳ Reading document {document:?}...");

                        match std::fs::read_to_string(document) {
                            Ok(mut document) if discord_chat => {
                                #[derive(serde::Deserialize)]
                                struct Chat {
                                    pub guild: Guild,
                                    pub channel: Channel,
                                    pub messages: Vec<Message>
                                }

                                #[derive(serde::Deserialize)]
                                struct Guild {
                                    pub name: String
                                }

                                #[derive(serde::Deserialize)]
                                struct Channel {
                                    // pub category: String,
                                    pub name: String,
                                    pub topic: Option<String>
                                }

                                #[derive(serde::Deserialize)]
                                struct Message {
                                    pub content: String,
                                    pub author: Author
                                }

                                #[derive(serde::Deserialize)]
                                struct Author {
                                    pub name: String,
                                    // pub nickname: String
                                }

                                if lowercase {
                                    document = document.to_lowercase();
                                }

                                let chat = match serde_json::from_str::<Chat>(&document) {
                                    Ok(chat) => chat,
                                    Err(err) => {
                                        eprintln!("{}", format!("🧯 Failed to parse chat history: {err}").red());

                                        return Ok(());
                                    }
                                };

                                drop(document);

                                let chat_name = format!(
                                    "<server>{}</server><channel>#{}</channel><topic>{}</topic>",
                                    &chat.guild.name,
                                    &chat.channel.name,
                                    chat.channel.topic.as_deref().unwrap_or("")
                                );

                                let messages = if discord_last_n == 0 || discord_last_n >= chat.messages.len() {
                                    &chat.messages
                                } else {
                                    &chat.messages[chat.messages.len() - discord_last_n..]
                                };

                                println!("⏳ Inserting {} chat messages...", messages.len());

                                if discord_split_documents {
                                    for i in 0..messages.len() {
                                        let message = &messages[i];

                                        let prev_message = messages.get(i - 1)
                                            .map(|message| message.content.as_str())
                                            .unwrap_or("");

                                        let document = Document::default()
                                            .with_name(&chat_name)
                                            .with_input(prev_message)
                                            .with_output(&message.content)
                                            .with_context(format!("<author>@{}</author>", message.author.name));

                                        if let Err(err) = database.insert(&document) {
                                            eprintln!("{}", format!("🧯 Failed to insert document: {err}").red());

                                            return Ok(());
                                        }
                                    }
                                }

                                else {
                                    let document = messages.iter()
                                        .map(|message| format!(
                                            "<message>@{}: {}</message>",
                                            message.author.name,
                                            message.content
                                        ))
                                        .fold(String::new(), |acc, message| acc + &message);

                                    let document = Document::new(document)
                                        .with_name(&chat_name);

                                    if let Err(err) = database.insert(&document) {
                                        eprintln!("{}", format!("🧯 Failed to insert document: {err}").red());

                                        return Ok(());
                                    }
                                }

                                println!("{}", "✅ Documents inserted".green());
                            }

                            Ok(mut document) => {
                                if lowercase {
                                    document = document.to_lowercase();
                                }

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
