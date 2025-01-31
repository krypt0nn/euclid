# Euclid: Stupid Natural Language Framework

Simplified tool for hand-crafting small language models.

## Raw input data preparation

Every model starts with input data preparation. You should search for some texts in the internet
in plaintext format. More clean data you have - better the final model will be.

- `euclid raw create --path raw_dataset.db` - create new raw documents database.
- `euclid raw push --database raw_dataset.db --file input_file.txt` - insert given input file into the database.
- `echo 'Example text' | euclid raw stream --path raw_dataset.db` - insert stdin data into the database.

## Building tokenizer

Tokenizer is a small language model that converts your natural language text into array of vectors
called "word embeddings". Tokenizers exist separate from your language models and could easily be
re-used or downloaded pre-trained by other people. Tokenizers benefit a lot from having very large
input data because they can use it to learn new words and do more precise embeddings.

Word embeddings are vectors representing single word in text. They're constructed in a way that
each vector dimension means some feature that makes words closer or further from each other.
For example, words `cat = [0.9, 0.71]` and `dog = [0.1, 0.72]` have very different first dimension
but very close second - we suspect the second dimension here means "domestic animal", or something
similar to it. Embeddings are trained using neural network with 1 hidden layer. They can improve
language models quality by a lot by grouping different words into groups by their meaning.

- `euclid tokenizer create --path tokenizer.db` - create new tokenizer with provided vector dimensions.
- `euclid tokenizer fit --dataset raw_dataset.db --path tokenizer.db` - fit tokenizer language model on provided raw dataset.
- `echo 'Example text' | euclid stream --path tokenizer.db` - return embeddings for text in stdin.
