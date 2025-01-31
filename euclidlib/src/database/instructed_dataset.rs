use std::path::Path;

use rusqlite::Connection;

#[derive(Debug)]
pub struct Database {
    connection: Connection
}

impl Database {
    /// Open raw dataset database with given cache size.
    /// Negative number means sqlite pages (1024 bytes), positive - bytes.
    pub fn open(path: impl AsRef<Path>, cache_size: i64) -> anyhow::Result<Self> {
        let connection = Connection::open(path)?;

        connection.execute(&format!("PRAGMA cache_size = {cache_size};"), ())?;

        connection.execute_batch("
            CREATE TABLE IF NOT EXISTS dataset (
                id      INTEGER NOT NULL,
                input   INTEGER NOT NULL,
                context INTEGER NOT NULL,
                output  INTEGER NOT NULL,

                PRIMARY KEY (id),

                FOREIGN KEY (input)   REFERENCES messages (message_id),
                FOREIGN KEY (context) REFERENCES messages (message_id),
                FOREIGN KEY (output)  REFERENCES messages (message_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER NOT NULL,
                token_id   INTEGER NOT NULL,

                PRIMARY KEY (message_id, token_id)
            );
        ")?;

        Ok(Self {
            connection
        })
    }

    /// Create new document from the given text.
    pub fn insert(&self, content: impl ToString) -> anyhow::Result<i64> {
        self.connection.prepare_cached("INSERT INTO documents (content) VALUES (?1)")?
            .execute([content.to_string()])?;

        Ok(self.connection.last_insert_rowid())
    }

    /// Get iterator of dataset documents and push them
    /// into the given callback.
    ///
    /// Return amount of read documents.
    pub fn for_each(&self, mut callback: impl FnMut(i64, String) -> anyhow::Result<()>) -> anyhow::Result<u64> {
        let mut read = 0;

        self.connection.prepare_cached("SELECT id, content FROM documents ORDER BY id ASC")?
            .query_map((), |row| {
                row.get::<_, i64>(0).and_then(|id| {
                    row.get::<_, String>(1).map(|content| (id, content))
                })
            })?
            .try_for_each(|row| {
                let (id, content) = row?;

                read += 1;

                callback(id, content)
            })
            .map(|_| read)
    }
}
