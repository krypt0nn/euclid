use clap::Parser;
use colorful::Colorful;

pub mod cli;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    if let Err(err) = cli::CLI::parse().execute() {
        eprintln!("{}", format!("🧯 An error occured: {err}").red());
    }
}
