#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use clap::Parser;
use colorful::Colorful;

use euclidlib::prelude::*;

pub mod cli;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    // if let Err(err) = cli::CLI::parse().execute() {
    //     eprintln!("{}", format!("🧯 An error occured: {err}").red());
    // }

    let token_hello = [1.0, 0.0, 0.0, 0.0, 0.0];
    let token_chat  = [0.0, 1.0, 0.0, 0.0, 0.0];
    let token_how   = [0.0, 0.0, 1.0, 0.0, 0.0];
    let token_are   = [0.0, 0.0, 0.0, 1.0, 0.0];
    let token_you   = [0.0, 0.0, 0.0, 0.0, 1.0];

    let text = [token_hello, token_chat, token_how, token_are, token_you].into_iter().flatten().collect::<Vec<_>>();

    let mut model = LSTMGenerator::<10, 5, f32>::random();

    let mut backpropagation = Backpropagation::<{ LSTMGenerator::<10, 5, f32>::PARAMS }, f32>::default();
    let mut device = CPUDevice::default();

    for _ in 0..1000 {
        backpropagation.timestep(|mut policy| {
            model.train(&text, 5, &mut policy, &mut device);
        });
    }

    let long_memory = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let short_memory = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let input = [
        1.0, 0.0, 0.0, 0.0, 0.0, // hello
        0.0, 1.0, 0.0, 0.0, 0.0  // chat
    ];

    let (_, _, output) = model.forward(long_memory, short_memory, input, &device);

    dbg!(output);
}
