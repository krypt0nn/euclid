use crate::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EncodedToken(i64);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DecodedToken(String);

pub struct Tokenizer {
    database: TokensDb,
    strip_punctuation: bool
}

impl Tokenizer {
    #[inline]
    pub fn in_database(database: TokensDb) -> Self {
        Self {
            database,
            strip_punctuation: false
        }
    }

    #[inline]
    pub fn with_strip_punctuation(&mut self, strip_punctuation: bool) -> &mut Self {
        self.strip_punctuation = strip_punctuation;

        self
    }

    /// Get iterator of separate words from the given string.
    pub fn decompose<'input>(&'input self, input: &'input str) -> impl Iterator<Item = &'input str> {
        input.split_whitespace()
            .map(|word| {
                let mut word = word.trim();

                if self.strip_punctuation {
                    word = word.trim_matches(|c: char| c.is_ascii_punctuation());
                }

                word
            })
            .filter(|word| !word.is_empty())
    }

    #[inline]
    /// Tokenize given string using underlying database.
    pub fn tokenize(&self, input: impl AsRef<str>) -> anyhow::Result<Vec<i64>> {
        self.decompose(input.as_ref())
            .map(|word| self.database.encode(word))
            .collect()
    }

    #[inline]
    /// Try to decode given token into a word.
    pub fn detokenize(&self, token: i64) -> anyhow::Result<String> {
        self.database.decode(token)
    }
}
