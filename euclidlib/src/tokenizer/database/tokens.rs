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

            CREATE INDEX idx_tokens_value on tokens (value);

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
        ")?;

        Ok(Self {
            connection
        })
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
}

#[test]
fn test_tokens_database() -> anyhow::Result<()> {
    let _ = std::fs::remove_file("tokens_database.db");

    let db = Database::open("tokens_database.db", 4096)?;

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

    let _ = std::fs::remove_file("tokens_database.db");

    Ok(())
}
