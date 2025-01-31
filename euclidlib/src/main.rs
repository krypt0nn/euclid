#![feature(generic_const_items, generic_const_exprs)]
#![allow(incomplete_features)]

pub mod document;
pub mod neural_network;
pub mod tokenizer;

pub mod token;
pub mod message;
pub mod instruction;
pub mod database;

pub mod prelude {
    pub use super::neural_network::prelude::*;
    pub use super::tokenizer::prelude::*;

    pub use super::document::Document;

    pub use super::token::*;
    pub use super::message::*;
    pub use super::instruction::*;
    pub use super::database::prelude::*;
}

fn main() {
    use std::collections::HashSet;

    use crate::prelude::*;

    let document = Document::new("
        In data analysis, cosine similarity is a measure of similarity between two non-zero vectors defined in an inner product space.
        Cosine similarity is the cosine of the angle between the vectors; that is, it is the dot product of the vectors divided by the
        product of their lengths. It follows that the cosine similarity does not depend on the magnitudes of the vectors, but only on
        their angle. The cosine similarity always belongs to the interval [ − 1 , 1 ] . For example, two proportional vectors have a cosine
        similarity of 1, two orthogonal vectors have a similarity of 0, and two opposite vectors have a similarity of -1. In some contexts,
        the component values of the vectors cannot be negative, in which case the cosine similarity is bounded in [ 0 , 1 ].

        For example, in information retrieval and text mining, each word is assigned a different coordinate and a document is represented by
        the vector of the numbers of occurrences of each word in the document. Cosine similarity then gives a useful measure of how similar two
        documents are likely to be, in terms of their subject matter, and independently of the length of the documents.[1]

        The technique is also used to measure cohesion within clusters in the field of data mining.[2]

        One advantage of cosine similarity is its low complexity, especially for sparse vectors: only the non-zero coordinates need to be considered.

        Other names for cosine similarity include Orchini similarity and Tucker coefficient of congruence; the Otsuka–Ochiai similarity (see below)
        is cosine similarity applied to binary data.[3]

        Another effective proxy for cosine distance can be obtained by L2 normalisation of the vectors, followed by the application of normal Euclidean
        distance. Using this technique each term in each vector is first divided by the magnitude of the vector, yielding a vector of unit length.
        Then the Euclidean distance over the end-points of any two vectors is a proper metric which gives the same ordering as the cosine distance
        (a monotonic transformation of Euclidean distance; see below) for any comparison of vectors, and furthermore avoids the potentially expensive
        trigonometric operations required to yield a proper metric. Once the normalisation has occurred, the vector space can be used with the full range
        of techniques available to any Euclidean space, notably standard dimensionality reduction techniques. This normalised form distance is often used
        within many deep learning algorithms.

        The most noteworthy property of cosine similarity is that it reflects a relative, rather than absolute, comparison of the individual vector
        dimensions. For any positive constant a and vector V, the vectors V and aV are maximally similar. The measure is thus most appropriate for data
        where frequency is more important than absolute values; notably, term frequency in documents. However more recent metrics with a grounding in
        information theory, such as Jensen–Shannon, SED, and triangular divergence have been shown to have improved semantics in at least some contexts. [14]

        A soft cosine or (\"soft\" similarity) between two vectors considers similarities between pairs of features.[19] The traditional cosine similarity
        considers the vector space model (VSM) features as independent or completely different, while the soft cosine measure proposes considering the
        similarity of features in VSM, which help generalize the concept of cosine (and soft cosine) as well as the idea of (soft) similarity.

        For example, in the field of natural language processing (NLP) the similarity among features is quite intuitive. Features such as words,
        n-grams, or syntactic n-grams[20] can be quite similar, though formally they are considered as different features in the VSM. For example,
        words \"play\" and \"game\" are different words and thus mapped to different points in VSM; yet they are semantically related. In case of
        n-grams or syntactic n-grams, Levenshtein distance can be applied (in fact, Levenshtein distance can be applied to words as well).

        For calculating soft cosine, the matrix s is used to indicate similarity between features. It can be calculated through Levenshtein distance,
        WordNet similarity, or other similarity measures. Then we just multiply by this matrix.

        The time complexity of this measure is quadratic, which makes it applicable to real-world tasks. Note that the complexity can be reduced
        to subquadratic.[21] An efficient implementation of such soft cosine similarity is included in the Gensim open source library.
    ");

    let word_tokens = DocumentsParser::new(true).parse(&document);

    let unique_tokens = HashSet::<String>::from_iter(word_tokens.clone());

    let numeric_tokens = word_tokens.iter()
        .cloned()
        .map(|word| {
            unique_tokens.iter()
                .position(|token| token == &word)
                .unwrap_or_default()
        })
        .collect::<Vec<usize>>();

    const RADIUS: usize = 4;
    const EMBEDDING: usize = 4;

    let mut loss_input = [0; RADIUS * 2];
    let loss_output = unique_tokens.iter().position(|token| token == &word_tokens[RADIUS]).unwrap_or_default();

    for i in 0..RADIUS {
        loss_input[i] = unique_tokens.iter().position(|token| token == &word_tokens[i]).unwrap_or_default();
    }

    for i in RADIUS..RADIUS * 2 {
        loss_input[i] = unique_tokens.iter().position(|token| token == &word_tokens[i + 1]).unwrap_or_default();
    }

    let cosine     = unique_tokens.iter().position(|token| token == "cosine").unwrap_or_default();
    let similarity = unique_tokens.iter().position(|token| token == "similarity").unwrap_or_default();
    let noteworthy = unique_tokens.iter().position(|token| token == "noteworthy").unwrap_or_default();

    let mut model = WordEmbeddingsModel::<2048, EMBEDDING, f64>::random();

    let mut backpropagation = Backpropagation::default()
        .with_warmup_duration(100)
        .with_cycle_period(100)
        .with_cycle_radius(0.00005)
        .with_learn_rate(0.00001);

    let now = std::time::Instant::now();

    for i in 0..300 {
        backpropagation.timestep(|mut policy| {
            model.train::<RADIUS>(&numeric_tokens, &mut policy);
        });

        if i % 20 == 0 && i != 0 {
            let loss = model.loss(&loss_input, loss_output);

            println!("loss = {loss:.8}, elapsed = {} seconds", now.elapsed().as_secs());

            if i % 100 == 0 && i != 0 {
                let cosine     = model.get_embedding(cosine);
                let similarity = model.get_embedding(similarity);
                let noteworthy = model.get_embedding(noteworthy);

                println!();

                println!("embedding(cosine)     = {cosine:?}");
                println!("embedding(similarity) = {similarity:?}");
                println!("embedding(noteworthy) = {noteworthy:?}");

                println!();

                println!("distance(cosine,     similarity) = {}", model.distance(&cosine, &similarity));
                println!("distance(cosine,     noteworthy) = {}", model.distance(&cosine, &noteworthy));
                println!("distance(similarity, noteworthy) = {}", model.distance(&similarity, &noteworthy));

                println!();
            }

            if loss < 0.5 {
                break;
            }
        }
    }

    // let mut table = String::from("token;x;y\n");

    // for (i, token) in unique_tokens.iter().enumerate() {
    //     if token.chars().next().unwrap().is_ascii_alphanumeric() {
    //         let [x, y] = model.get_embedding(i);

    //         table += &format!("\"{token}\";{x};{y}\n");
    //     }
    // }

    // std::fs::write("table.csv", table).unwrap();
}
