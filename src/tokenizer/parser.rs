use crate::prelude::*;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Parser {
    /// Convert all words to lowercase.
    pub lowercase: bool
}

impl Parser {
    pub const DOCUMENT_OPEN_TAG: &'static str = "<document>";
    pub const DOCUMENT_CLOSE_TAG: &'static str = "</document>";

    pub const NAME_OPEN_TAG: &'static str = "<name>";
    pub const NAME_CLOSE_TAG: &'static str = "</name>";

    pub const INPUT_OPEN_TAG: &'static str = "<input>";
    pub const INPUT_CLOSE_TAG: &'static str = "</input>";

    pub const CONTEXT_OPEN_TAG: &'static str = "<context>";
    pub const CONTEXT_CLOSE_TAG: &'static str = "</context>";

    pub const OUTPUT_OPEN_TAG: &'static str = "<output>";
    pub const OUTPUT_CLOSE_TAG: &'static str = "</output>";

    #[inline]
    pub fn new(lowercase: bool) -> Self {
        Self {
            lowercase
        }
    }

    /// Return vector of separate words and symbols (tokens) from the given document.
    pub fn parse(&self, document: &Document) -> Vec<String> {
        fn parse_text(text: &str, lowercase: bool) -> Vec<String> {
            let mut tokens = Vec::new();

            let mut i = 0;
            let mut j = 0;

            let text: Vec<char> = if lowercase {
                text.to_lowercase().chars().collect()
            } else {
                text.chars().collect()
            };

            let n = text.len();

            while j < n {
                // Continue collecting alpha-numerics (literal values built from letters and numbers).
                if text[j].is_alphanumeric() {
                    j += 1;
                }

                // Skip whitespaces.
                else if text[j].is_whitespace() {
                    // Store the word before whitespace.
                    if i < j {
                        tokens.push(text[i..j].iter().collect());
                    }

                    // Skip all the following whitespaces as well.
                    while j < n && text[j].is_whitespace() {
                        tokens.push(text[j].to_string());

                        j += 1;
                    }

                    // Set cursor to the whitespace's end.
                    i = j;
                }

                // Store special symbol (non-alpha-numeric value).
                else {
                    // Store the word before the symbol.
                    if i < j {
                        tokens.push(text[i..j].iter().collect());
                    }

                    // Store the symbol.
                    tokens.push(text[j].to_string());

                    // Update cursors.
                    j += 1;
                    i = j;
                }
            }

            // Store remaining word.
            if i < j {
                tokens.push(text[i..j].iter().collect());
            }

            tokens
        }

        let mut name_tokens = parse_text(&document.name, self.lowercase);
        let mut input_tokens = parse_text(&document.input, self.lowercase);
        let mut context_tokens = parse_text(&document.context, self.lowercase);
        let mut output_tokens = parse_text(&document.output, self.lowercase);

        let mut tokens = Vec::with_capacity(name_tokens.len() + input_tokens.len() + context_tokens.len() + output_tokens.len() + 10);

        // <document>
        tokens.push(Self::DOCUMENT_OPEN_TAG.to_string());

        // <name>...</name>
        tokens.push(Self::NAME_OPEN_TAG.to_string());
        tokens.append(&mut name_tokens);
        tokens.push(Self::NAME_CLOSE_TAG.to_string());

        // <input>...</input>
        tokens.push(Self::INPUT_OPEN_TAG.to_string());
        tokens.append(&mut input_tokens);
        tokens.push(Self::INPUT_CLOSE_TAG.to_string());

        // <context>...</context>
        tokens.push(Self::CONTEXT_OPEN_TAG.to_string());
        tokens.append(&mut context_tokens);
        tokens.push(Self::CONTEXT_CLOSE_TAG.to_string());

        // <output>...</output>
        tokens.push(Self::OUTPUT_OPEN_TAG.to_string());
        tokens.append(&mut output_tokens);
        tokens.push(Self::OUTPUT_CLOSE_TAG.to_string());

        // </document>
        tokens.push(Self::DOCUMENT_CLOSE_TAG.to_string());

        tokens
    }

    /// Try to reconstruct document from the given tokens slice.
    ///
    /// Return `None` if provided tokens have invalid format.
    pub fn join(&self, tokens: &[String]) -> Option<Document> {
        let document = Document::default();

        // <document>...</document>
        if tokens.first().map(String::as_str) != Some(Self::DOCUMENT_OPEN_TAG) || tokens.last().map(String::as_str) != Some(Self::DOCUMENT_CLOSE_TAG) {
            return None;
        }

        let tokens = &tokens[1..tokens.len() - 1];

        fn parse_section<'a>(tokens: &'a [String], open: &'static str, close: &'static str) -> Option<(String, &'a [String])> {
            // <tag>...</tag>
            if tokens.first().map(String::as_str) != Some(open) {
                return None;
            }

            let mut i = 1;
            let n = tokens.len();

            while i < n {
                if tokens[i] == close {
                    return Some((tokens[1..i].concat(), &tokens[i + 1..]));
                }

                i += 1;
            }

            None
        }

        let (name,    tokens) = parse_section(tokens, Self::NAME_OPEN_TAG,    Self::NAME_CLOSE_TAG)?;
        let (input,   tokens) = parse_section(tokens, Self::INPUT_OPEN_TAG,   Self::INPUT_CLOSE_TAG)?;
        let (context, tokens) = parse_section(tokens, Self::CONTEXT_OPEN_TAG, Self::CONTEXT_CLOSE_TAG)?;
        let (output, _tokens) = parse_section(tokens, Self::OUTPUT_OPEN_TAG,  Self::OUTPUT_CLOSE_TAG)?;

        Some(document.with_name(name)
            .with_input(input)
            .with_context(context)
            .with_output(output))
    }
}

#[test]
fn test_document_tokenizer() {
    let document = Document::new("Example document")
        .with_name("With some example name")
        .with_input("With <very> =special11- #@%\"<-input->\"!")
        .with_context(""); // and empty context

    let tokens = Parser::default()
        .parse(&document);

    assert_eq!(tokens, &[
        "<document>",
            "<name>", "With", " ", "some", " ", "example", " ", "name", "</name>",
            "<input>", "With", " ", "<", "very", ">", " ", "=", "special11", "-", " ", "#", "@", "%", "\"", "<", "-", "input", "-", ">", "\"", "!", "</input>",
            "<context>", "</context>",
            "<output>", "Example", " ", "document", "</output>",
        "</document>"
    ]);

    let detokenized = Parser::default()
        .join(&tokens)
        .unwrap();

    assert_eq!(document, detokenized);
}
