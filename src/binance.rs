use serde::Deserialize;
use std::fmt;
use std::mem::size_of;
use zerocopy::byteorder::LittleEndian;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

type U16LE = zerocopy::byteorder::U16<LittleEndian>;
type U32LE = zerocopy::byteorder::U32<LittleEndian>;
type I64LE = zerocopy::byteorder::I64<LittleEndian>;

const TEMPLATE_TRADES: u16 = 10000;
const TEMPLATE_BEST_BID_ASK: u16 = 10001;
const TEMPLATE_DEPTH_SNAPSHOT: u16 = 10002;
const TEMPLATE_DEPTH_DIFF: u16 = 10003;

#[derive(Debug, Clone)]
pub enum BinanceEvent {
    Trades(TradesStreamEvent),
    BestBidAsk(BestBidAskStreamEvent),
    DepthSnapshot(DepthSnapshotStreamEvent),
    DiffBookDepth(DepthDiffStreamEvent),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TradesStreamEvent {
    pub event_time_us: i64,
    pub transact_time_us: i64,
    pub price_exponent: i8,
    pub qty_exponent: i8,
    pub trades: Vec<Trade>,
    pub symbol: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trade {
    pub id: i64,
    pub price_m: i64,
    pub qty_m: i64,
    /// Buyer was the resting side, so the aggressor sold.
    pub is_buyer_maker: bool,
}

#[derive(Debug, Clone)]
pub struct BestBidAskStreamEvent {
    pub event_time_us: i64,
    pub book_update_id: i64,
    pub price_exponent: i8,
    pub qty_exponent: i8,
    pub best_bid_price_m: i64,
    pub best_bid_qty_m: i64,
    pub best_ask_price_m: i64,
    pub best_ask_qty_m: i64,
    pub best_bid: f64,
    pub best_bid_qty: f64,
    pub best_ask: f64,
    pub best_ask_qty: f64,
    pub symbol: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DepthSnapshotStreamEvent {
    pub event_time_us: i64,
    pub book_update_id: i64,
    pub price_exponent: i8,
    pub qty_exponent: i8,
    pub bids: Vec<DepthLevel>,
    pub asks: Vec<DepthLevel>,
    pub symbol: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DepthDiffStreamEvent {
    pub event_time_us: i64,
    pub first_update_id: i64,
    pub last_update_id: i64,
    pub price_exponent: i8,
    pub qty_exponent: i8,
    pub bids: Vec<DepthLevel>,
    pub asks: Vec<DepthLevel>,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DepthSnapshotResponse {
    pub last_update_id: i64,
    pub bids: Vec<DepthLevel>,
    pub asks: Vec<DepthLevel>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub struct DepthLevel {
    #[serde(deserialize_with = "from_str_to_f64")]
    pub price: f64,
    #[serde(deserialize_with = "from_str_to_f64")]
    pub qty: f64,
    #[serde(skip_deserializing)]
    pub price_m: Option<i64>,
    #[serde(skip_deserializing)]
    pub qty_m: Option<i64>,
}

impl DepthLevel {
    pub fn new(price: f64, qty: f64, price_m: Option<i64>, qty_m: Option<i64>) -> Self {
        Self {
            price,
            qty,
            price_m,
            qty_m,
        }
    }
}

impl BinanceEvent {
    pub fn from_sbe(payload: &[u8]) -> Option<Self> {
        let (header, rest) = MessageHeader::read_from_prefix(payload).ok()?;
        let block_length = header.block_length.get() as usize;

        match header.template_id.get() {
            TEMPLATE_TRADES => {
                let (block, rest) = read_root_block::<TradesBlock>(rest, block_length)?;
                let (trades, rest) = parse_trades(rest)?;
                let symbol = parse_symbol(rest)?;

                Some(Self::Trades(TradesStreamEvent {
                    event_time_us: block.event_time.get(),
                    transact_time_us: block.transact_time.get(),
                    price_exponent: block.price_exponent,
                    qty_exponent: block.qty_exponent,
                    trades,
                    symbol,
                }))
            }
            TEMPLATE_BEST_BID_ASK => {
                let (block, rest) = read_root_block::<BestBidAskBlock>(rest, block_length)?;
                let symbol = parse_symbol(rest)?;
                let price_exponent = block.price_exponent;
                let qty_exponent = block.qty_exponent;
                let best_bid_price_m = block.bid_price.get();
                let best_bid_qty_m = block.bid_qty.get();
                let best_ask_price_m = block.ask_price.get();
                let best_ask_qty_m = block.ask_qty.get();

                Some(Self::BestBidAsk(BestBidAskStreamEvent {
                    event_time_us: block.event_time.get(),
                    book_update_id: block.book_update_id.get(),
                    price_exponent,
                    qty_exponent,
                    best_bid_price_m,
                    best_bid_qty_m,
                    best_ask_price_m,
                    best_ask_qty_m,
                    best_bid: mantissa64_to_f64(best_bid_price_m, price_exponent),
                    best_bid_qty: mantissa64_to_f64(best_bid_qty_m, qty_exponent),
                    best_ask: mantissa64_to_f64(best_ask_price_m, price_exponent),
                    best_ask_qty: mantissa64_to_f64(best_ask_qty_m, qty_exponent),
                    symbol,
                }))
            }
            TEMPLATE_DEPTH_SNAPSHOT => {
                let (block, rest) = read_root_block::<DepthSnapshotBlock>(rest, block_length)?;
                let (bids, rest) =
                    parse_depth_levels(rest, block.price_exponent, block.qty_exponent)?;
                let (asks, rest) =
                    parse_depth_levels(rest, block.price_exponent, block.qty_exponent)?;
                let symbol = parse_symbol(rest)?;

                Some(Self::DepthSnapshot(DepthSnapshotStreamEvent {
                    event_time_us: block.event_time.get(),
                    book_update_id: block.book_update_id.get(),
                    price_exponent: block.price_exponent,
                    qty_exponent: block.qty_exponent,
                    bids,
                    asks,
                    symbol,
                }))
            }
            TEMPLATE_DEPTH_DIFF => {
                let (block, rest) = read_root_block::<DepthDiffBlock>(rest, block_length)?;
                let (bids, rest) =
                    parse_depth_levels(rest, block.price_exponent, block.qty_exponent)?;
                let (asks, rest) =
                    parse_depth_levels(rest, block.price_exponent, block.qty_exponent)?;
                let symbol = parse_symbol(rest)?;

                Some(Self::DiffBookDepth(DepthDiffStreamEvent {
                    event_time_us: block.event_time.get(),
                    first_update_id: block.first_book_update_id.get(),
                    last_update_id: block.last_book_update_id.get(),
                    price_exponent: block.price_exponent,
                    qty_exponent: block.qty_exponent,
                    bids,
                    asks,
                    symbol,
                }))
            }
            _ => None,
        }
    }
}

fn read_root_block<T: FromBytes>(payload: &[u8], block_length: usize) -> Option<(T, &[u8])> {
    let (block, rest) = payload.split_at_checked(block_length)?;
    let parsed = T::read_from_prefix(block).ok()?.0;
    Some((parsed, rest))
}

fn parse_depth_levels(payload: &[u8], price_e: i8, qty_e: i8) -> Option<(Vec<DepthLevel>, &[u8])> {
    let (group_header, rest) = GroupSize16Encoding::read_from_prefix(payload).ok()?;
    let entry_count = group_header.num_in_group.get() as usize;
    let block_length = group_header.block_length.get() as usize;

    if entry_count == 0 {
        return Some((Vec::new(), rest));
    }

    if block_length < size_of::<DepthEntry>() {
        return None;
    }

    let entries_len = entry_count.checked_mul(block_length)?;
    let (entries, rest) = rest.split_at_checked(entries_len)?;
    let mut levels = Vec::with_capacity(entry_count);

    for chunk in entries.chunks_exact(block_length) {
        let entry = DepthEntry::read_from_prefix(chunk).ok()?.0;
        let price_m = entry.price.get();
        let qty_m = entry.qty.get();
        levels.push(DepthLevel::new(
            mantissa64_to_f64(price_m, price_e),
            mantissa64_to_f64(qty_m, qty_e),
            Some(price_m),
            Some(qty_m),
        ));
    }

    Some((levels, rest))
}

fn parse_trades(payload: &[u8]) -> Option<(Vec<Trade>, &[u8])> {
    let (group_header, rest) = GroupSizeEncoding::read_from_prefix(payload).ok()?;
    let entry_count = group_header.num_in_group.get() as usize;
    let block_length = group_header.block_length.get() as usize;

    if entry_count == 0 {
        return Some((Vec::new(), rest));
    }
    if block_length < size_of::<TradeEntry>() {
        return None;
    }

    let entries_len = entry_count.checked_mul(block_length)?;
    let (entries, rest) = rest.split_at_checked(entries_len)?;
    let trades = entries
        .chunks_exact(block_length)
        .map(|chunk| {
            let entry = TradeEntry::read_from_prefix(chunk).ok()?.0;
            Some(Trade {
                id: entry.id.get(),
                price_m: entry.price.get(),
                qty_m: entry.qty.get(),
                is_buyer_maker: entry.is_buyer_maker != 0,
            })
        })
        .collect::<Option<Vec<_>>>()?;

    Some((trades, rest))
}

fn parse_symbol(payload: &[u8]) -> Option<String> {
    let mut offset = 0usize;
    let symbol = get_var_str_u8(payload, &mut offset);
    if symbol.is_empty() {
        return None;
    }
    let symbol = std::str::from_utf8(symbol).ok()?;
    Some(symbol.to_owned())
}

fn get_var_str_u8<'a>(data: &'a [u8], offset: &mut usize) -> &'a [u8] {
    if *offset + 1 > data.len() {
        return &[];
    }
    let len = data[*offset] as usize;
    *offset += 1;
    if *offset + len > data.len() {
        return &[];
    }
    let result = &data[*offset..*offset + len];
    *offset += len;
    result
}

fn from_str_to_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct F64Visitor;

    impl<'de> serde::de::Visitor<'de> for F64Visitor {
        type Value = Option<f64>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a string containing an f64 number")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if value.is_empty() {
                Ok(None)
            } else {
                Ok(Some(value.parse::<f64>().map_err(E::custom)?))
            }
        }
    }

    deserializer
        .deserialize_str(F64Visitor)
        .map(|value| value.unwrap_or(0.0))
}

#[inline(always)]
pub fn mantissa64_to_f64(mantissa: i64, exponent: i8) -> f64 {
    const POW10_POS: [f64; 23] = [
        1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
        1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
    ];
    const POW10_NEG: [f64; 23] = [
        1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13,
        1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22,
    ];

    let idx = exponent.unsigned_abs() as usize;
    if idx >= POW10_POS.len() {
        return if exponent > 0 {
            if mantissa >= 0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        } else {
            0.0
        };
    }

    if exponent >= 0 {
        mantissa as f64 * POW10_POS[idx]
    } else {
        mantissa as f64 * POW10_NEG[idx]
    }
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct MessageHeader {
    block_length: U16LE,
    template_id: U16LE,
    #[allow(dead_code)]
    schema_id: U16LE,
    #[allow(dead_code)]
    version: U16LE,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct GroupSize16Encoding {
    block_length: U16LE,
    num_in_group: U16LE,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct GroupSizeEncoding {
    block_length: U16LE,
    num_in_group: U32LE,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct TradesBlock {
    event_time: I64LE,
    transact_time: I64LE,
    price_exponent: i8,
    qty_exponent: i8,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct TradeEntry {
    id: I64LE,
    price: I64LE,
    qty: I64LE,
    is_buyer_maker: u8,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct BestBidAskBlock {
    event_time: I64LE,
    book_update_id: I64LE,
    price_exponent: i8,
    qty_exponent: i8,
    bid_price: I64LE,
    bid_qty: I64LE,
    ask_price: I64LE,
    ask_qty: I64LE,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct DepthSnapshotBlock {
    event_time: I64LE,
    book_update_id: I64LE,
    price_exponent: i8,
    qty_exponent: i8,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct DepthDiffBlock {
    event_time: I64LE,
    first_book_update_id: I64LE,
    last_book_update_id: I64LE,
    price_exponent: i8,
    qty_exponent: i8,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct DepthEntry {
    price: I64LE,
    qty: I64LE,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "left={left}, right={right}");
    }

    fn push_u16(buf: &mut Vec<u8>, value: u16) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_i64(buf: &mut Vec<u8>, value: i64) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_i8(buf: &mut Vec<u8>, value: i8) {
        buf.push(value as u8);
    }

    fn push_symbol(buf: &mut Vec<u8>, value: &str) {
        buf.push(value.len() as u8);
        buf.extend_from_slice(value.as_bytes());
    }

    fn push_header(buf: &mut Vec<u8>, block_length: u16, template_id: u16) {
        push_u16(buf, block_length);
        push_u16(buf, template_id);
        push_u16(buf, 0);
        push_u16(buf, 0);
    }

    #[test]
    fn parses_best_bid_ask_from_sbe() {
        let mut payload = Vec::new();
        push_header(&mut payload, 50, TEMPLATE_BEST_BID_ASK);
        push_i64(&mut payload, 999);
        push_i64(&mut payload, 77);
        push_i8(&mut payload, -1);
        push_i8(&mut payload, -2);
        push_i64(&mut payload, 123);
        push_i64(&mut payload, 456);
        push_i64(&mut payload, 124);
        push_i64(&mut payload, 654);
        push_symbol(&mut payload, "ETHUSDT");

        let event = BinanceEvent::from_sbe(&payload);

        match event {
            Some(BinanceEvent::BestBidAsk(book)) => {
                assert_eq!(book.event_time_us, 999);
                assert_eq!(book.book_update_id, 77);
                assert_eq!(book.best_bid_price_m, 123);
                assert_eq!(book.best_bid_qty_m, 456);
                assert_eq!(book.best_ask_price_m, 124);
                assert_eq!(book.best_ask_qty_m, 654);
                assert_close(book.best_bid, 12.3);
                assert_close(book.best_bid_qty, 4.56);
                assert_close(book.best_ask, 12.4);
                assert_close(book.best_ask_qty, 6.54);
                assert_eq!(book.symbol.as_str(), "ETHUSDT");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn parses_trades_from_sbe() {
        let mut payload = Vec::new();
        push_header(&mut payload, 18, TEMPLATE_TRADES);
        push_i64(&mut payload, 1_000);
        push_i64(&mut payload, 999);
        push_i8(&mut payload, -2);
        push_i8(&mut payload, -3);
        push_u16(&mut payload, 25);
        payload.extend_from_slice(&2u32.to_le_bytes());
        for (id, price, qty, maker) in [(7, 10_050, 1_500, 1u8), (8, 10_051, 250, 0u8)] {
            push_i64(&mut payload, id);
            push_i64(&mut payload, price);
            push_i64(&mut payload, qty);
            payload.push(maker);
        }
        push_symbol(&mut payload, "BTCUSDT");

        match BinanceEvent::from_sbe(&payload) {
            Some(BinanceEvent::Trades(event)) => {
                assert_eq!(event.event_time_us, 1_000);
                assert_eq!(event.transact_time_us, 999);
                assert_eq!(event.price_exponent, -2);
                assert_eq!(event.qty_exponent, -3);
                assert_eq!(
                    event.trades,
                    vec![
                        Trade {
                            id: 7,
                            price_m: 10_050,
                            qty_m: 1_500,
                            is_buyer_maker: true,
                        },
                        Trade {
                            id: 8,
                            price_m: 10_051,
                            qty_m: 250,
                            is_buyer_maker: false,
                        },
                    ]
                );
                assert_eq!(event.symbol.as_str(), "BTCUSDT");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn parses_depth_snapshot_from_sbe() {
        let mut payload = Vec::new();
        push_header(&mut payload, 18, TEMPLATE_DEPTH_SNAPSHOT);
        push_i64(&mut payload, 5_000);
        push_i64(&mut payload, 321);
        push_i8(&mut payload, -1);
        push_i8(&mut payload, -3);
        push_u16(&mut payload, 16);
        push_u16(&mut payload, 2);
        push_i64(&mut payload, 101);
        push_i64(&mut payload, 10_000);
        push_i64(&mut payload, 100);
        push_i64(&mut payload, 11_000);
        push_u16(&mut payload, 16);
        push_u16(&mut payload, 1);
        push_i64(&mut payload, 102);
        push_i64(&mut payload, 9_000);
        push_symbol(&mut payload, "SOLUSDT");

        let event = BinanceEvent::from_sbe(&payload);

        match event {
            Some(BinanceEvent::DepthSnapshot(snapshot)) => {
                assert_eq!(snapshot.event_time_us, 5_000);
                assert_eq!(snapshot.book_update_id, 321);
                assert_eq!(snapshot.bids.len(), 2);
                assert_eq!(snapshot.asks.len(), 1);
                assert_eq!(snapshot.bids[0].price_m, Some(101));
                assert_eq!(snapshot.bids[0].qty_m, Some(10_000));
                assert_eq!(snapshot.asks[0].price_m, Some(102));
                assert_eq!(snapshot.asks[0].qty_m, Some(9_000));
                assert_close(snapshot.bids[0].price, 10.1);
                assert_close(snapshot.bids[0].qty, 10.0);
                assert_close(snapshot.asks[0].price, 10.2);
                assert_close(snapshot.asks[0].qty, 9.0);
                assert_eq!(snapshot.symbol.as_str(), "SOLUSDT");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn parses_depth_diff_from_sbe() {
        let mut payload = Vec::new();
        push_header(&mut payload, 26, TEMPLATE_DEPTH_DIFF);
        push_i64(&mut payload, 6_000);
        push_i64(&mut payload, 400);
        push_i64(&mut payload, 401);
        push_i8(&mut payload, -2);
        push_i8(&mut payload, -1);
        push_u16(&mut payload, 16);
        push_u16(&mut payload, 1);
        push_i64(&mut payload, 12_345);
        push_i64(&mut payload, 50);
        push_u16(&mut payload, 16);
        push_u16(&mut payload, 1);
        push_i64(&mut payload, 12_355);
        push_i64(&mut payload, 80);
        push_symbol(&mut payload, "XRPUSDT");

        let event = BinanceEvent::from_sbe(&payload);

        match event {
            Some(BinanceEvent::DiffBookDepth(diff)) => {
                assert_eq!(diff.event_time_us, 6_000);
                assert_eq!(diff.first_update_id, 400);
                assert_eq!(diff.last_update_id, 401);
                assert_close(diff.bids[0].price, 123.45);
                assert_close(diff.bids[0].qty, 5.0);
                assert_close(diff.asks[0].price, 123.55);
                assert_close(diff.asks[0].qty, 8.0);
                assert_eq!(diff.symbol.as_str(), "XRPUSDT");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn rejects_truncated_symbol_payload() {
        let mut payload = Vec::new();
        push_header(&mut payload, 50, TEMPLATE_BEST_BID_ASK);
        push_i64(&mut payload, 999);
        push_i64(&mut payload, 77);
        push_i8(&mut payload, -1);
        push_i8(&mut payload, -2);
        push_i64(&mut payload, 123);
        push_i64(&mut payload, 456);
        push_i64(&mut payload, 124);
        push_i64(&mut payload, 654);
        payload.push(7);
        payload.extend_from_slice(b"ETH");

        assert!(BinanceEvent::from_sbe(&payload).is_none());
    }
}
