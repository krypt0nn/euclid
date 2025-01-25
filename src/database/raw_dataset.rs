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
            CREATE TABLE IF NOT EXISTS documents (
                id      INTEGER NOT NULL,
                content TEXT    NOT NULL,

                PRIMARY KEY (id)
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

#[test]
fn test_raw_dateset() -> anyhow::Result<()> {
    let _ = std::fs::remove_file("raw_dataset.db");

    let db = Database::open("raw_dataset.db", 4096)?;

    db.insert("Test document 1")?;
    db.insert("Test document 2")?;
    db.insert("Test document 3")?;

    let read = db.for_each(|id, document| {
        assert_eq!(document, format!("Test document {id}"));

        Ok(())
    })?;

    assert_eq!(read, 3);

    let _ = std::fs::remove_file("raw_dataset.db");

    Ok(())
}
