use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::PathBuf;
use zstd::stream::read::Decoder as ZstdDecoder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawRecord {
    pub timestamp_ns: i64,
    pub tag: u8,
}

pub struct RawSbeReader {
    file_paths: VecDeque<PathBuf>,
    current_reader: Option<BufReader<ZstdDecoder<'static, BufReader<File>>>>,
}

impl RawSbeReader {
    pub fn new<I>(paths: I) -> Self
    where
        I: IntoIterator<Item = PathBuf>,
    {
        Self {
            file_paths: paths.into_iter().collect(),
            current_reader: None,
        }
    }

    pub fn next_record(&mut self, payload: &mut Vec<u8>) -> io::Result<Option<RawRecord>> {
        loop {
            if self.current_reader.is_none() {
                if let Some(path) = self.file_paths.pop_front() {
                    let file = File::open(path)?;
                    let decoder = ZstdDecoder::with_buffer(BufReader::new(file))?;
                    self.current_reader = Some(BufReader::new(decoder));
                } else {
                    return Ok(None);
                }
            }

            let reader = self
                .current_reader
                .as_mut()
                .expect("reader just initialized");

            let mut ts_buf = [0u8; 8];
            if let Err(error) = reader.read_exact(&mut ts_buf) {
                if error.kind() == io::ErrorKind::UnexpectedEof {
                    self.current_reader = None;
                    continue;
                }
                return Err(error);
            }

            let mut tag_buf = [0u8; 1];
            reader.read_exact(&mut tag_buf)?;

            let mut len_buf = [0u8; 4];
            reader.read_exact(&mut len_buf)?;

            let payload_len = u32::from_le_bytes(len_buf) as usize;
            payload.resize(payload_len, 0);
            reader.read_exact(payload)?;

            return Ok(Some(RawRecord {
                timestamp_ns: i64::from_le_bytes(ts_buf),
                tag: tag_buf[0],
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};
    use zstd::stream::write::Encoder as ZstdEncoder;

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("valid clock")
            .as_nanos();
        std::env::temp_dir().join(format!("sstoikov_{name}_{}_{}", std::process::id(), nanos))
    }

    #[test]
    fn reads_records_across_multiple_files() -> io::Result<()> {
        let dir = temp_dir("raw_reader");
        std::fs::create_dir_all(&dir)?;

        for (index, name) in ["a.zst", "b.zst"].iter().enumerate() {
            let file = File::create(dir.join(name))?;
            let mut encoder = ZstdEncoder::new(file, 1)?;

            encoder.write_all(&(1000 + index as i64).to_le_bytes())?;
            encoder.write_all(b"S")?;
            encoder.write_all(&(4u32).to_le_bytes())?;
            encoder.write_all(b"test")?;

            encoder.write_all(&(2000 + index as i64).to_le_bytes())?;
            encoder.write_all(b"R")?;
            encoder.write_all(&(3u32).to_le_bytes())?;
            encoder.write_all(b"abc")?;
            encoder.finish()?;
        }

        let mut reader = RawSbeReader::new(vec![dir.join("a.zst"), dir.join("b.zst")]);
        let mut payload = Vec::new();

        let record = reader.next_record(&mut payload)?.expect("record 1");
        assert_eq!(record.timestamp_ns, 1000);
        assert_eq!(record.tag, b'S');
        assert_eq!(payload, b"test");

        let record = reader.next_record(&mut payload)?.expect("record 2");
        assert_eq!(record.timestamp_ns, 2000);
        assert_eq!(record.tag, b'R');
        assert_eq!(payload, b"abc");

        let record = reader.next_record(&mut payload)?.expect("record 3");
        assert_eq!(record.timestamp_ns, 1001);
        assert_eq!(record.tag, b'S');
        assert_eq!(payload, b"test");

        let record = reader.next_record(&mut payload)?.expect("record 4");
        assert_eq!(record.timestamp_ns, 2001);
        assert_eq!(record.tag, b'R');
        assert_eq!(payload, b"abc");

        assert!(reader.next_record(&mut payload)?.is_none());

        std::fs::remove_dir_all(dir)?;
        Ok(())
    }
}
