use std::path::Path;

use rusqlite::Connection;

use crate::prelude::*;

#[derive(Debug)]
/// SQLite database for storing raw documents.
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
            CREATE TABLE IF NOT EXISTS documents (
                id      INTEGER NOT NULL,
                name    BLOB    NOT NULL,
                input   BLOB    NOT NULL,
                context BLOB    NOT NULL,
                output  BLOB    NOT NULL,

                PRIMARY KEY (id)
            );
        ")?;

        Ok(Self {
            connection
        })
    }

    /// Insert document to the database.
    pub fn insert(&self, document: &Document) -> anyhow::Result<i64> {
        self.connection.prepare_cached("INSERT INTO documents (name, input, context, output) VALUES (?1, ?2, ?3, ?4)")?
            .execute([
                lz4_flex::compress_prepend_size(document.name.as_bytes()),
                lz4_flex::compress_prepend_size(document.input.as_bytes()),
                lz4_flex::compress_prepend_size(document.context.as_bytes()),
                lz4_flex::compress_prepend_size(document.output.as_bytes())
            ])?;

        Ok(self.connection.last_insert_rowid())
    }

    /// Iterate over all the documents stored in the database.
    ///
    /// Return amount of read documents.
    pub fn for_each(&self, mut callback: impl FnMut(i64, Document) -> anyhow::Result<()>) -> anyhow::Result<u64> {
        let mut read = 0;

        self.connection.prepare_cached("SELECT id, name, input, context, output FROM documents ORDER BY id ASC")?
            .query_map((), |row| {
                let id      = row.get::<_, i64>(0)?;
                let name    = row.get::<_, Vec<u8>>(1)?;
                let input   = row.get::<_, Vec<u8>>(2)?;
                let context = row.get::<_, Vec<u8>>(3)?;
                let output  = row.get::<_, Vec<u8>>(4)?;

                Ok((id, name, input, context, output))
            })?
            .map(|row| -> anyhow::Result<_> {
                let (id, name, input, context, output) = row?;

                let name    = lz4_flex::decompress_size_prepended(&name)?;
                let input   = lz4_flex::decompress_size_prepended(&input)?;
                let context = lz4_flex::decompress_size_prepended(&context)?;
                let output  = lz4_flex::decompress_size_prepended(&output)?;

                Ok((id, name, input, context, output))
            })
            .try_for_each(|row| {
                let (id, name, input, context, output) = row?;

                read += 1;

                callback(id, Document {
                    name: String::from_utf8(name)?,
                    input: String::from_utf8(input)?,
                    context: String::from_utf8(context)?,
                    output: String::from_utf8(output)?
                })
            })
            .map(|_| read)
    }
}

#[test]
fn test_documents_database() -> anyhow::Result<()> {
    let _ = std::fs::remove_file("documents_database.db");

    let db = Database::open("documents_database.db", 4096)?;

    db.insert(&Document::new("Test document 1"))?;
    db.insert(&Document::new("Test document 2"))?;
    db.insert(&Document::new("Test document 3"))?;

    let read = db.for_each(|id, document| {
        assert_eq!(document.output, format!("Test document {id}"));

        Ok(())
    })?;

    assert_eq!(read, 3);

    let _ = std::fs::remove_file("documents_database.db");

    Ok(())
}
