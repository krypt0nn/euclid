use std::path::Path;

use rusqlite::Connection;

#[derive(Debug)]
pub struct Database {
    connection: Connection
}

impl Database {
    /// Open tokens dataset database with given cache size.
    /// Negative number means sqlite pages (1024 bytes), positive - bytes.
    pub fn open(path: impl AsRef<Path>, cache_size: i64) -> anyhow::Result<Self> {
        let connection = Connection::open(path)?;

        connection.execute(&format!("PRAGMA cache_size = {cache_size};"), ())?;

        connection.execute_batch("
            CREATE TABLE IF NOT EXISTS tokens (
                id   INTEGER NOT NULL,
                text TEXT    NOT NULL,

                PRIMARY KEY (id)
            );
        ")?;

        Ok(Self {
            connection
        })
    }

    /// Try to get token from the given word.
    pub fn encode(&self, word: impl AsRef<str>) -> anyhow::Result<i64> {
        let word = word.as_ref();

        let row = self.connection.prepare_cached("SELECT id FROM tokens WHERE text = ?1")?
            .query_row([word], |row| row.get(0));

        if let Ok(token) = row {
            return Ok(token);
        }

        self.connection.prepare_cached("INSERT INTO tokens (text) VALUES (?1)")?
            .execute([word])?;

        Ok(self.connection.last_insert_rowid())
    }

    /// Try getting word from the given token.
    pub fn decode(&self, token: i64) -> anyhow::Result<String> {
        let word = self.connection.prepare_cached("SELECT text FROM tokens WHERE id = ?1")?
            .query_row([token], |row| row.get(0))?;

        Ok(word)
    }
}

#[test]
fn test_tokens_dateset() -> anyhow::Result<()> {
    let _ = std::fs::remove_file("tokens.db");

    let db = Database::open("tokens.db", 4096)?;

    let hello = db.encode("Hello,")?;
    let world = db.encode("World!")?;

    assert_ne!(hello, world);

    assert_eq!(db.decode(hello)?, "Hello,");
    assert_eq!(db.decode(world)?, "World!");

    let _ = std::fs::remove_file("tokens.db");

    Ok(())
}
