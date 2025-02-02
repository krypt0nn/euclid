use std::path::Path;

use rusqlite::Connection;

use crate::prelude::*;

#[derive(Debug)]
/// SQLite database for storing tokens, embeddings and word embeddings model.
pub struct Database {
    connection: Connection
}

impl Database {
    /// Open database with given cache size.
    /// Negative number means sqlite pages (1024 bytes), positive - bytes.
    pub fn open(path: impl AsRef<Path>, cache_size: i64) -> rusqlite::Result<Self> {
        let connection = Connection::open(path)?;

        connection.execute(&format!("PRAGMA cache_size = {cache_size};"), ())?;

        connection.execute_batch("
            CREATE TABLE IF NOT EXISTS tokens (
                id    INTEGER NOT NULL,
                value TEXT UNIQUE NOT NULL,

                PRIMARY KEY (id)
            );

            CREATE INDEX IF NOT EXISTS idx_tokens_value on tokens (value);

            CREATE TABLE IF NOT EXISTS embeddings (
                token_id  INTEGER NOT NULL,
                embedding BLOB NOT NULL,

                PRIMARY KEY (token_id),
                FOREIGN KEY (token_id) REFERENCES tokens (id)
            );

            CREATE TABLE IF NOT EXISTS encoder_neurons (
                id     INTEGER NOT NULL,
                params BLOB NOT NULL,

                PRIMARY KEY (id)
            );

            CREATE TABLE IF NOT EXISTS decoder_neurons (
                id     INTEGER NOT NULL,
                params BLOB NOT NULL,

                PRIMARY KEY (id)
            );

            INSERT OR IGNORE INTO tokens (id, value) VALUES (0, '');
            INSERT OR IGNORE INTO embeddings (token_id, embedding) VALUES (0, '');
        ")?;

        Ok(Self {
            connection
        })
    }

    /// Iterate over all tokens stored in the database.
    ///
    /// Return amount of read tokens.
    pub fn for_each(&self, mut callback: impl FnMut(i64, String) -> anyhow::Result<()>) -> anyhow::Result<u64> {
        let mut tokens = 0;

        self.connection.prepare_cached("SELECT id, value FROM tokens ORDER BY id ASC")?
            .query_map([], move |row| {
                let id = row.get::<_, i64>(0)?;
                let token = row.get::<_, String>(1)?;

                Ok((id, token))
            })?
            .try_for_each(|row| {
                let (id, token) = row?;

                tokens += 1;

                callback(id, token)
            })?;

        Ok(tokens)
    }

    /// Query token from the database.
    ///
    /// Guaranteed to return `Ok(None)` if token is not stored.
    pub fn query_token(&self, token: impl AsRef<str>) -> rusqlite::Result<Option<i64>> {
        let id = self.connection.prepare_cached("SELECT id FROM tokens WHERE value = ?1")?
            .query_row([token.as_ref()], |row| row.get::<_, i64>(0));

        match id {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => Err(err)
        }
    }

    /// Insert token to the database.
    ///
    /// Return id of inserted token.
    pub fn insert_token(&self, token: impl AsRef<str>) -> rusqlite::Result<i64> {
        let Some(id) = self.query_token(token.as_ref())? else {
            self.connection.prepare_cached("INSERT INTO tokens (value) VALUES (?1)")?
                .execute([token.as_ref()])?;

            return Ok(self.connection.last_insert_rowid());
        };

        Ok(id)
    }

    /// Query token embedding from the database.
    ///
    /// Guaranteed to return `Ok(None)` if token is not stored.
    /// Can return `Err` if `F` is different from the stored type.
    pub fn query_embedding<F: Float>(&self, token_id: i64) -> anyhow::Result<Option<Vec<F>>> where [(); F::BYTES]: Sized {
        let embedding = self.connection.prepare_cached("SELECT embedding FROM embeddings WHERE token_id = ?1")?
            .query_row([token_id], |row| row.get::<_, Vec<u8>>(0));

        match embedding {
            Ok(embedding_bytes) => {
                let n = embedding_bytes.len();

                if n % F::BYTES != 0 {
                    anyhow::bail!("Trying to query token {token_id} embedding with different float type");
                }

                let mut embedding = Vec::with_capacity(n / F::BYTES);
                let mut bytes = [0; F::BYTES];

                let mut k = 0;

                while k < n {
                    bytes.copy_from_slice(&embedding_bytes[k..k + F::BYTES]);

                    embedding.push(F::from_bytes(&bytes));

                    k += F::BYTES;
                }

                Ok(Some(embedding))
            }

            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => anyhow::bail!(err)
        }
    }

    /// Insert token embedding to the database.
    pub fn insert_embedding<F: Float>(&self, token_id: i64, embedding: &[F]) -> rusqlite::Result<()> where [(); F::BYTES]: Sized {
        let mut embedding_bytes = vec![0; embedding.len() * F::BYTES];

        for (i, float) in embedding.iter().enumerate() {
            embedding_bytes[i * F::BYTES..(i + 1) * F::BYTES].copy_from_slice(&float.to_bytes());
        }

        self.connection.prepare_cached("INSERT OR REPLACE INTO embeddings (token_id, embedding) VALUES (?1, ?2)")?
            .execute((token_id, embedding_bytes))?;

        Ok(())
    }

    /// Find token in the database with the closest to given embedding.
    ///
    /// Guaranteed to return some value unless the database is empty.
    ///
    /// `N` specifies amount of dimensions which will be respected by the
    /// cosine similarity function. Make sure to use equal or larger value
    /// than the size of stored embeddings.
    pub fn find_token<const N: usize, F: Float>(&self, embedding: &[F]) -> anyhow::Result<Option<i64>> where [(); F::BYTES]: Sized {
        let mut rows = self.connection.prepare_cached("SELECT token_id FROM embeddings")?
            .query_map((), |row| row.get::<_, i64>(0))?
            .map(|id| -> anyhow::Result<_> {
                let id = id?;

                Ok((id, self.query_embedding::<F>(id)?))
            })
            .flat_map(|row| {
                match row {
                    Ok((token, Some(embedding))) => Some(Ok((token, embedding))),
                    Ok((_, None)) => None,
                    Err(err) => Some(Err(err))
                }
            })
            .map(|row| -> anyhow::Result<_> {
                let (token, token_embedding) = row?;

                Ok((token, cosine_similarity::<N, F>(&token_embedding, embedding)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let Some(mut closest_token) = rows.pop() else {
            return Ok(None);
        };

        for (token, similarity) in rows {
            if similarity >= 1.0 {
                return Ok(Some(token));
            }

            if similarity > closest_token.1 {
                closest_token = (token, similarity);
            }
        }

        Ok(Some(closest_token.0))
    }

    #[allow(unused_braces)]
    /// Save given word embeddings model into the database.
    pub fn save_model<
        const TOKENS_NUM: usize,
        const EMBEDDING_SIZE: usize,
        F: Float
    >(&self, model: &GenericWordEmbeddingsModel<TOKENS_NUM, EMBEDDING_SIZE, F>) -> anyhow::Result<()>
    where
        [(); { (TOKENS_NUM + 1) * F::BYTES }]: Sized,
        [(); { (EMBEDDING_SIZE + 1) * F::BYTES }]: Sized
    {
        self.connection.execute_batch("DELETE FROM encoder_neurons; DELETE FROM decoder_neurons;")?;

        let mut query = self.connection.prepare_cached("INSERT INTO encoder_neurons (params) VALUES (?1)")?;

        for neuron in model.encoder_decoder.encoder.neurons() {
            query.execute([neuron.to_bytes()])?;
        }

        let mut query = self.connection.prepare_cached("INSERT INTO decoder_neurons (params) VALUES (?1)")?;

        for neuron in model.encoder_decoder.decoder.neurons() {
            query.execute([neuron.to_bytes()])?;
        }

        Ok(())
    }

    #[allow(unused_braces)]
    /// Save given dynamically sized word embeddings model into the database.
    pub fn save_dynamic_model<F: Float>(&self, model: &WordEmbeddingsModel<F>) -> anyhow::Result<()>
    where
        // This incredible where statement is needed to fix const generics which are BROKEN!!!!!!!!

        [(); { 1024    + 1 } * F::BYTES]: Sized,
        [(); { 4096    + 1 } * F::BYTES]: Sized,
        [(); { 16384   + 1 } * F::BYTES]: Sized,
        [(); { 65536   + 1 } * F::BYTES]: Sized,
        [(); { 262144  + 1 } * F::BYTES]: Sized,
        [(); { 1048576 + 1 } * F::BYTES]: Sized,

        [(); { 32   + 1 } * F::BYTES]: Sized,
        [(); { 64   + 1 } * F::BYTES]: Sized,
        [(); { 128  + 1 } * F::BYTES]: Sized,
        [(); { 256  + 1 } * F::BYTES]: Sized,
        [(); { 512  + 1 } * F::BYTES]: Sized,
        [(); { 1024 + 1 } * F::BYTES]: Sized
    {
        match model {
            WordEmbeddingsModel::Tiny(model)   => self.save_model::<1024,    32,   F>(model),
            WordEmbeddingsModel::Small(model)  => self.save_model::<4096,    64,   F>(model),
            WordEmbeddingsModel::Medium(model) => self.save_model::<16384,   128,  F>(model),
            WordEmbeddingsModel::Large(model)  => self.save_model::<65536,   256,  F>(model),
            WordEmbeddingsModel::Huge(model)   => self.save_model::<262144,  512,  F>(model),
            WordEmbeddingsModel::Giant(model)  => self.save_model::<1048576, 1024, F>(model)
        }
    }

    /// Return input tokens number and embeddings size
    /// of the stored model.
    ///
    /// Guaranteed to return `Ok(None)` if the model is not saved.
    pub fn saved_model_params(&self) -> rusqlite::Result<Option<(usize, usize)>> {
        let mut query = self.connection.prepare("SELECT
            (SELECT COUNT(encoder_neurons.id) FROM encoder_neurons) as encoder_size,
            (SELECT COUNT(decoder_neurons.id) FROM decoder_neurons) as decoder_size")?;

        let sizes = query.query_row([], |row| {
            let encoder_size = row.get::<_, usize>(0)?;
            let decoder_size = row.get::<_, usize>(1)?;

            Ok((encoder_size, decoder_size))
        });

        match sizes {
            Ok((0, _)) | Ok((_, 0)) => Ok(None),
            Ok((encoder_size, decoder_size)) => Ok(Some((decoder_size, encoder_size))),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => Err(err)
        }
    }

    /// Try to load model from the database.
    ///
    /// This method required you to specify *exact* size of the model.
    ///
    /// Guaranteed to return `Ok(None)` if the model is not saved.
    pub fn load_model<
        const TOKENS_NUM: usize,
        const EMBEDDING_SIZE: usize,
        F: Float
    >(&self) -> anyhow::Result<Option<GenericWordEmbeddingsModel<TOKENS_NUM, EMBEDDING_SIZE, F>>>
    where
        [(); { TOKENS_NUM + 1 } * F::BYTES]: Sized,
        [(); { EMBEDDING_SIZE + 1 } * F::BYTES]: Sized
    {
        // Prepare model-related buffers.

        let mut encoder_layer = unsafe {
            alloc_fixed_heap_array::<Neuron<TOKENS_NUM, F>, EMBEDDING_SIZE>()
                .expect("Failed to allocate memory for word embeddings encoder layer neurons")
        };

        let mut decoder_layer = unsafe {
            alloc_fixed_heap_array::<Neuron<EMBEDDING_SIZE, F>, TOKENS_NUM>()
                .expect("Failed to allocate memory for word embeddings decoder layer neurons")
        };

        let mut encoder_neuron = unsafe {
            alloc_fixed_heap_array::<u8, { (TOKENS_NUM + 1) * F::BYTES }>()
                .expect("Failed to allocate memory for a single encoder neuron")
        };

        let mut decoder_neuron = unsafe {
            alloc_fixed_heap_array::<u8, { (EMBEDDING_SIZE + 1) * F::BYTES }>()
                .expect("Failed to allocate memory for a single decoder neuron")
        };

        // Check model existance and validate its sizes.

        let Some((inputs, embedding)) = self.saved_model_params()? else {
            return Ok(None);
        };

        if inputs != TOKENS_NUM {
            anyhow::bail!("Trying to load model with input size {inputs} as {TOKENS_NUM}");
        }

        if embedding != EMBEDDING_SIZE {
            anyhow::bail!("Trying to load model with embedding size {embedding} as {EMBEDDING_SIZE}");
        }

        // Fill encoder layer buffer.

        self.connection.prepare("SELECT params FROM encoder_neurons ORDER BY id ASC")?
            .query_map([], |row| {
                encoder_neuron.copy_from_slice(&row.get::<_, Vec<u8>>(0)?);

                let neuron = Neuron::<TOKENS_NUM, F>::from_bytes(&encoder_neuron);

                Ok(neuron)
            })?
            .enumerate()
            .try_for_each(|(i, neuron)| -> rusqlite::Result<()> {
                encoder_layer[i] = neuron?;

                Ok(())
            })?;

        drop(encoder_neuron);

        // Fill decoder layer buffer.

        self.connection.prepare("SELECT params FROM decoder_neurons ORDER BY id ASC")?
            .query_map([], |row| {
                decoder_neuron.copy_from_slice(&row.get::<_, Vec<u8>>(0)?);

                let neuron = Neuron::<EMBEDDING_SIZE, F>::from_bytes(&decoder_neuron);

                Ok(neuron)
            })?
            .enumerate()
            .try_for_each(|(i, neuron)| -> rusqlite::Result<()> {
                decoder_layer[i] = neuron?;

                Ok(())
            })?;

        drop(decoder_neuron);

        Ok(Some(GenericWordEmbeddingsModel {
            encoder_decoder: EncoderDecoder::from_layers(
                Layer::from_neurons(encoder_layer),
                Layer::from_neurons(decoder_layer)
            )
        }))
    }

    /// Try to load dynamically sized model from the database.
    /// This method expects one of standard supported generic models.
    ///
    /// Guaranteed to return `Ok(None)` if the model is not saved.
    pub fn load_dynamic_model<F: Float>(&self) -> anyhow::Result<Option<WordEmbeddingsModel<F>>>
    where
        // This incredible where statement is needed to fix const generics which are BROKEN!!!!!!!!

        [(); { 1024    + 1 } * F::BYTES]: Sized,
        [(); { 4096    + 1 } * F::BYTES]: Sized,
        [(); { 16384   + 1 } * F::BYTES]: Sized,
        [(); { 65536   + 1 } * F::BYTES]: Sized,
        [(); { 262144  + 1 } * F::BYTES]: Sized,
        [(); { 1048576 + 1 } * F::BYTES]: Sized,

        [(); { 32   + 1 } * F::BYTES]: Sized,
        [(); { 64   + 1 } * F::BYTES]: Sized,
        [(); { 128  + 1 } * F::BYTES]: Sized,
        [(); { 256  + 1 } * F::BYTES]: Sized,
        [(); { 512  + 1 } * F::BYTES]: Sized,
        [(); { 1024 + 1 } * F::BYTES]: Sized
    {
        let Some((inputs, embedding)) = self.saved_model_params()? else {
            return Ok(None);
        };

        match (inputs, embedding) {
            (1024,    32)   => Ok(self.load_model::<1024,    32,   F>()?.map(WordEmbeddingsModel::from)),
            (4096,    64)   => Ok(self.load_model::<4096,    64,   F>()?.map(WordEmbeddingsModel::from)),
            (16384,   128)  => Ok(self.load_model::<16384,   128,  F>()?.map(WordEmbeddingsModel::from)),
            (65536,   256)  => Ok(self.load_model::<65536,   256,  F>()?.map(WordEmbeddingsModel::from)),
            (262144,  512)  => Ok(self.load_model::<262144,  512,  F>()?.map(WordEmbeddingsModel::from)),
            (1048576, 1024) => Ok(self.load_model::<1048576, 1024, F>()?.map(WordEmbeddingsModel::from)),

            _ => anyhow::bail!("Trying to load non-standard model with {inputs} inputs and {embedding} embedding size")
        }
    }
}

#[test]
fn test_tokens_database() -> anyhow::Result<()> {
    let _ = std::fs::remove_file("tokens_database.db");

    let db = Database::open("tokens_database.db", 4096)?;

    assert!(db.query_token("hello")?.is_none());
    assert!(db.query_embedding::<f32>(1)?.is_none());
    assert!(db.find_token::<1, f32>(&[1.0])?.is_none());

    db.insert_token("hello")?;
    db.insert_token("world")?;

    assert_eq!(db.query_token("hello")?, Some(1));
    assert_eq!(db.query_token("world")?, Some(2));

    db.insert_embedding::<f32>(1, &[1.0, 2.0, 3.0])?;
    db.insert_embedding::<f32>(2, &[1.0, 2.0, 4.0])?;

    assert_eq!(db.query_embedding::<f32>(1)?.as_deref(), Some([1.0, 2.0, 3.0].as_slice()));
    assert_eq!(db.query_embedding::<f32>(2)?.as_deref(), Some([1.0, 2.0, 4.0].as_slice()));

    assert_eq!(db.find_token::<3, f32>(&[1.0, 2.0, 3.0])?, Some(1));
    assert_eq!(db.find_token::<3, f32>(&[1.0, 2.0, 4.0])?, Some(2));

    assert_eq!(db.find_token::<3, f32>(&[1.0, 2.0, 3.4])?, Some(1));
    assert_eq!(db.find_token::<3, f32>(&[1.0, 2.0, 3.6])?, Some(2));

    assert!(db.saved_model_params()?.is_none());

    let model = GenericWordEmbeddingsModel::<16, 4, f32>::random();

    db.save_model(&model)?;

    assert_eq!(db.saved_model_params()?, Some((16, 4)));

    let loaded_model = db.load_model::<16, 4, f32>()?.unwrap();

    for i in 0..4 {
        assert_eq!(
            loaded_model.encoder_decoder.encoder.neurons[i].weights,
            model.encoder_decoder.encoder.neurons[i].weights
        );

        assert_eq!(
            loaded_model.encoder_decoder.encoder.neurons[i].bias,
            model.encoder_decoder.encoder.neurons[i].bias
        );
    }

    for i in 0..16 {
        assert_eq!(
            loaded_model.encoder_decoder.decoder.neurons[i].weights,
            model.encoder_decoder.decoder.neurons[i].weights
        );

        assert_eq!(
            loaded_model.encoder_decoder.decoder.neurons[i].bias,
            model.encoder_decoder.decoder.neurons[i].bias
        );
    }

    let _ = std::fs::remove_file("tokens_database.db");

    Ok(())
}
