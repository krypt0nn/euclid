#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
/// Individual text corpus used for useful data extraction.
pub struct Document {
    /// Name of the document. This can be a title of a book,
    /// article, name of a friends group chat, and so on.
    ///
    /// Theoretically this could be used by the model to generate
    /// different style responses.
    ///
    /// You can keep this field empty.
    pub name: String,

    /// Input text which was used to generate the document's output.
    ///
    /// For documents downloaded from the internet for
    /// model training this field can remain empty.
    pub input: String,

    /// Context sentence which explains environment used
    /// to produce output text from the document's input.
    ///
    /// Theoretically this could be used by the model to generate
    /// different responses for the same input based on some
    /// external info like daytime or the person they talk to.
    ///
    /// For documents downloaded from the internet for
    /// model training this field can remain empty.
    pub context: String,

    /// Output text. Actual content of the document.
    pub output: String
}

impl Document {
    #[inline]
    /// Create new document from its content.
    ///
    /// This method will keep document's name, input
    /// and context fields empty and fill only its output.
    pub fn new(content: impl ToString) -> Self {
        Self {
            name: String::new(),
            input: String::new(),
            context: String::new(),
            output: content.to_string()
        }
    }

    #[inline]
    /// Change document's name.
    pub fn with_name(mut self, name: impl ToString) -> Self {
        self.name = name.to_string();

        self
    }

    #[inline]
    /// Change document's input.
    pub fn with_input(mut self, input: impl ToString) -> Self {
        self.input = input.to_string();

        self
    }

    #[inline]
    /// Change document's context.
    pub fn with_context(mut self, context: impl ToString) -> Self {
        self.context = context.to_string();

        self
    }

    #[inline]
    /// Change document's output.
    pub fn with_output(mut self, output: impl ToString) -> Self {
        self.output = output.to_string();

        self
    }
}
