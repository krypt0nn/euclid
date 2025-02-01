use std::collections::HashSet;

use euclidlib::prelude::*;

pub mod cli;

fn main() {
    let document = Document::new(std::fs::read_to_string("PoliticalEconomy_truncated.txt").unwrap());

    let word_tokens = DocumentsParser::new(true).parse(&document);

    let unique_tokens = HashSet::<String>::from_iter(word_tokens.clone())
        .into_iter()
        .collect::<Vec<_>>();

    println!("Tokenized into {} unique tokens", unique_tokens.len());

    let numeric_tokens = word_tokens.iter()
        .cloned()
        .map(|word| {
            unique_tokens.iter()
                .position(|token| token == &word)
                .unwrap_or_default()
        })
        .collect::<Vec<usize>>();

    type GenericModel = GenericWordEmbeddingsModel::<2048, 16, f64>;

    const RADIUS: usize = 8;

    let mut loss_input = [0; RADIUS * 2];
    let loss_output = unique_tokens.iter().position(|token| token == &word_tokens[RADIUS]).unwrap_or_default();

    for i in 0..RADIUS {
        loss_input[i] = unique_tokens.iter().position(|token| token == &word_tokens[i]).unwrap_or_default();
    }

    for i in RADIUS..RADIUS * 2 {
        loss_input[i] = unique_tokens.iter().position(|token| token == &word_tokens[i + 1]).unwrap_or_default();
    }

    let political = unique_tokens.iter().position(|token| token == "political").unwrap_or_default();
    let economy   = unique_tokens.iter().position(|token| token == "economy").unwrap_or_default();
    let wealth    = unique_tokens.iter().position(|token| token == "wealth").unwrap_or_default();

    let mut model = WordEmbeddingsModel::from_generic(GenericModel::random()).unwrap();

    let (tokens, embeddings, params) = model.params();

    println!("tokens = {tokens}, embeddings = {embeddings}, params = {params}\n");

    // let mut model = GenericModel::random();

    let mut backpropagation = Backpropagation::default()
        .with_warmup_duration(10)
        .with_cycle_period(20)
        .with_cycle_radius(0.0015)
        .with_learn_rate(0.0015);

    let now = std::time::Instant::now();

    for i in 0..500 {
        backpropagation.timestep(|mut policy| {
            model.train::<RADIUS>(&numeric_tokens, &mut policy);
        });

        let loss = model.loss(&loss_input, loss_output);

        println!("loss = {loss:.8}, elapsed = {} seconds", now.elapsed().as_secs());

        if i % 10 == 0 && i != 0 {
            let political = model.encode(political);
            let economy   = model.encode(economy);
            let wealth    = model.encode(wealth);

            println!();

            println!("embedding(political) = {political:?}");
            println!("embedding(economy)   = {economy:?}");
            println!("embedding(wealth)    = {wealth:?}");

            println!();

            println!("distance(political, economy) = {}", cosine_similarity::<64, _>(&political, &economy));
            println!("distance(political, wealth)  = {}", cosine_similarity::<64, _>(&political, &wealth));
            println!("distance(economy,   wealth)  = {}", cosine_similarity::<64, _>(&economy,   &wealth));

            println!();

            println!("decode(embedding(political)) = {:?}", unique_tokens.get(model.decode(&political)));
            println!("decode(embedding(economy))   = {:?}", unique_tokens.get(model.decode(&economy)));
            println!("decode(embedding(wealth))    = {:?}", unique_tokens.get(model.decode(&wealth)));

            println!();

            let mut table = String::from("token,x,y\n");

            for (i, token) in unique_tokens.iter().enumerate() {
                if token.chars().next().unwrap().is_ascii_alphanumeric() {
                    let embedding = model.encode(i);

                    table += &format!("\"{token}\",{},{}\n", embedding[0], embedding[1]);
                }
            }

            std::fs::write(format!("embeddings_{i}.csv"), table).unwrap();
        }
    }
}
