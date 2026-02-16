use serde::Deserialize;
use sstoikov_microprice::Level;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct DiffBookDepthEvent {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "U")]
    pub first_update_id: u64,
    #[serde(rename = "u")]
    pub final_update_id: u64,
    #[serde(rename = "pu")]
    pub last_update_id: Option<u64>,
    #[serde(rename = "b")]
    pub bids: Vec<RawLevel>,
    #[serde(rename = "a")]
    pub asks: Vec<RawLevel>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawLevel(String, String);

impl RawLevel {
    pub fn to_level(&self) -> Option<Level> {
        let price = self.0.parse().ok()?;
        let qty = self.1.parse().ok()?;
        Some(Level {
            price,
            quantity: qty,
        })
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct DepthUpdateEvent {
    pub stream: String,
    pub data: DiffBookDepthEvent,
}

#[derive(Debug, Default, Clone)]
pub struct LocalOrderbook {
    pub is_synced: bool,
    pub bids: BTreeMap<u64, Level>,
    pub asks: BTreeMap<u64, Level>,
    /// U
    pub first_update_id: u64,
    /// u
    pub final_update_id: u64,
    /// pu
    pub last_update_id: u64,
    pub inv_tick_size: f64,
}

impl LocalOrderbook {
    pub fn new(tick_size: f64) -> Self {
        Self {
            is_synced: false,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            first_update_id: 0,
            final_update_id: 0,
            last_update_id: 0,
            inv_tick_size: 1.0 / tick_size,
        }
    }

    #[inline]
    pub fn process_diff_book_depth(&mut self, event: DiffBookDepthEvent) {
        self.first_update_id = event.first_update_id;
        self.final_update_id = event.final_update_id;
        if let Some(lid) = event.last_update_id {
            self.last_update_id = lid;
        }

        // Bids
        for raw_level in &event.bids {
            if let Some(level) = raw_level.to_level() {
                let price_ticks = (level.price * self.inv_tick_size).round() as u64;
                if level.quantity == 0.0 {
                    self.bids.remove(&price_ticks);
                } else {
                    self.bids.insert(price_ticks, level);
                }
            }
        }

        // Asks
        for raw_level in &event.asks {
            if let Some(level) = raw_level.to_level() {
                let price_ticks = (level.price * self.inv_tick_size).round() as u64;
                if level.quantity == 0.0 {
                    self.asks.remove(&price_ticks);
                } else {
                    // For safe comparison/ordering in BTreeMap which sorts by Key (u64)
                    // Price is key.
                    self.asks.insert(price_ticks, level);
                }
            }
        }

        self.is_synced = true;
    }

    // Top N levels: Bids are highest price first, Asks are lowest price first.
    // BTreeMap iterates keys in ascending order.

    #[inline]
    pub fn get_bids(&self, n: usize) -> Vec<Level> {
        self.bids.values().rev().take(n).cloned().collect()
    }

    #[inline]
    pub fn get_asks(&self, n: usize) -> Vec<Level> {
        self.asks.values().take(n).cloned().collect()
    }

    // Construct simplified State for the crate logic
    pub fn get_state(&self, depth: usize) -> Option<sstoikov_microprice::OrderBookState> {
        if self.bids.is_empty() || self.asks.is_empty() {
            return None;
        }
        let bids = self.get_bids(depth);
        let asks = self.get_asks(depth);
        if bids.is_empty() || asks.is_empty() {
            return None;
        }
        Some(sstoikov_microprice::OrderBookState { bids, asks })
    }
}
