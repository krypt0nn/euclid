#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use std::collections::HashSet;

pub mod neural_network;

pub mod document;
pub mod token;
pub mod message;
pub mod instruction;
pub mod database;

pub mod prelude {
    pub use super::neural_network::prelude::*;

    pub use super::token::*;
    pub use super::message::*;
    pub use super::instruction::*;
    pub use super::database::prelude::*;
}

#[derive(Debug, Clone)]
struct WordVecModel<const EMBEDDING_SIZE: usize> {
    words: Vec<String>,
    vocabulary: HashSet<String>,

    // VOCABULARY_SIZE x EMBEDDING_SIZE
    input_layer_weights: Vec<[f64; EMBEDDING_SIZE]>,

    // EMBEDDING_SIZE x VOCABULARY_SIZE
    output_layer_weights: Vec<Vec<f64>>
}

impl<const EMBEDDING_SIZE: usize> WordVecModel<EMBEDDING_SIZE> {
    pub fn from_words(words: impl IntoIterator<Item = String>) -> Self {
        let mut words_vec = Vec::new();
        let mut vocabulary = HashSet::new();

        for word in words.into_iter() {
            words_vec.push(word.clone());
            vocabulary.insert(word);
        }

        let mut input_layer_weights = Vec::new();

        for _ in 0..vocabulary.len() {
            let mut weights = [0.0; EMBEDDING_SIZE];

            for i in 0..EMBEDDING_SIZE {
                weights[i] = fastrand::f64();
            }

            input_layer_weights.push(weights);
        }

        let mut output_layer_weights = Vec::new();

        #[allow(clippy::needless_range_loop)]
        for _ in 0..EMBEDDING_SIZE {
            let weights = (0..vocabulary.len())
                .map(|_| fastrand::f64())
                .collect::<Vec<_>>();

            output_layer_weights.push(weights);
        }

        Self {
            words: words_vec,
            vocabulary,
            input_layer_weights,
            output_layer_weights
        }
    }

    /// Perform single network training iteration,
    /// returning current error rate (higher is worse).
    pub fn train(&self, learning_rate: f64, mut radius: usize) -> f64 {
        let n = self.words.len();

        // Don't even start training if there's too few words for this.
        if n < 3 {
            return 1.0;
        }

        // Reduce radius by 1 while we can't fit words into it.
        while radius > 1 && radius * 2 + 1 > n {
            radius -= 1;
        }

        // Iterate through the input words, skipping first and last R.
        for i in radius..n - radius {
            // Get current target word.
            let target_word = &self.words[i];

            // Iterate over the words surrounding currently targeted one in radius R.
            // for j in i - radius..j + radius {
            //     // Skip current word itself.
            //     if i != j {
            //         let mut error = [0.0; EMBEDDING_SIZE];

            //         for i in 0..EMBEDDING_SIZE {
            //             error[i] = self.output_layer_weights[i][j].tanh();
            //         }

            //         // # Forward pass
            //         // hidden_layer = self.W1[target_index]
            //         // output_layer = self.W2.T @ hidden_layer
            //         // exp_scores = np.exp(output_layer)
            //         // probabilities = exp_scores / np.sum(exp_scores)

            //         // # Backward pass
            //         // target = np.zeros(len(self.vocab))
            //         // target[context_index] = 1  # One-hot encoding of the context word

            //         // # Calculate the error
            //         // error = probabilities - target

            //         // # Update weights
            //         // self.W1[target_index] -= self.learning_rate * error @ self.W2.T
            //         // self.W2[:, context_index] -= self.learning_rate * hidden_layer * error[context_index]
            //     }
            // }
        }

        1.0
    }
}

fn main() {
    use crate::prelude::*;

    // Make 2 identical neurons.
    let mut neuron_1 = Neuron64::linear();
    let mut neuron_2 = neuron_1.clone();

    // Make default backpropagation policy.
    let mut policy_1 = Backpropagation::default();

    // Policy with cyclic schedule and warmup period.
    let mut policy_2 = Backpropagation::default()
        .with_warmup_duration(10)
        .with_cycle_period(15)
        .with_cycle_radius(0.015);

    // let mut policy_2 = Backpropagation::explore(|policy| {
    //     let mut neuron = neuron_2.clone();

    //     for _ in 0..20 {
    //         neuron.backward(&[0.0, 1.0], 1.0, policy);
    //         neuron.backward(&[2.0, 0.0], 2.0, policy);
    //         neuron.backward(&[1.0, 1.0], 2.0, policy);
    //         neuron.backward(&[2.0, 1.0], 3.0, policy);
    //     }

    //     neuron.loss(neuron.forward(&[3.0, -1.0]), 2.0).as_f64()
    // });

    dbg!(&policy_1, &policy_2);

    for _ in 0..100 {
        neuron_1.backward(&[0.0, 1.0], 1.0, &mut policy_1);
        neuron_1.backward(&[2.0, 0.0], 2.0, &mut policy_1);
        neuron_1.backward(&[1.0, 1.0], 2.0, &mut policy_1);
        neuron_1.backward(&[2.0, 1.0], 3.0, &mut policy_1);

        neuron_2.backward(&[0.0, 1.0], 1.0, &mut policy_2);
        neuron_2.backward(&[2.0, 0.0], 2.0, &mut policy_2);
        neuron_2.backward(&[1.0, 1.0], 2.0, &mut policy_2);
        neuron_2.backward(&[2.0, 1.0], 3.0, &mut policy_2);

        let loss_1 = neuron_1.loss(neuron_1.forward(&[3.0, -1.0]), 2.0);
        let loss_2 = neuron_2.loss(neuron_2.forward(&[3.0, -1.0]), 2.0);

        println!("loss 1: {loss_1:.8}, loss 2: {loss_2:.8}");
    }

    dbg!(neuron_1, neuron_2);
}
