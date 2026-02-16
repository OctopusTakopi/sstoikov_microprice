// use std::collections::BTreeMap;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Level {
    pub price: f64,
    pub quantity: f64,
}

#[derive(Error, Debug)]
pub enum MicropriceError {
    #[error("Orderbook is empty or missing data on one side")]
    EmptyBook,
    #[error("Requested depth {depth} is not sufficient (bids: {bid_len}, asks: {ask_len})")]
    InsufficientDepth {
        depth: usize,
        bid_len: usize,
        ask_len: usize,
    },
    #[error("Zero liquidity at requested depth")]
    ZeroLiquidity,
}

#[derive(Debug, Clone)]
pub struct OrderBookState {
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
}

impl OrderBookState {
    /// Calculate the mid price (average of best bid and best ask).
    pub fn mid_price(&self) -> Result<f64, MicropriceError> {
        if self.bids.is_empty() || self.asks.is_empty() {
            return Err(MicropriceError::EmptyBook);
        }
        // Assuming bids sorted descending and asks ascending
        Ok((self.bids[0].price + self.asks[0].price) / 2.0)
    }

    /// Calculate the Order Book Imbalance at a specific depth.
    /// Imbalance = BidQty / (BidQty + AskQty)
    pub fn imbalance(&self, depth_level: usize) -> Result<f64, MicropriceError> {
        let bid_len = self.bids.len();
        let ask_len = self.asks.len();

        if bid_len < depth_level || ask_len < depth_level {
            return Err(MicropriceError::InsufficientDepth {
                depth: depth_level,
                bid_len,
                ask_len,
            });
        }

        let bid_qty: f64 = self.bids.iter().take(depth_level).map(|l| l.quantity).sum();
        let ask_qty: f64 = self.asks.iter().take(depth_level).map(|l| l.quantity).sum();

        let total = bid_qty + ask_qty;
        if total == 0.0 {
            Err(MicropriceError::ZeroLiquidity)
        } else {
            Ok(bid_qty / total)
        }
    }

    /// Calculate the Volume-Weighted Mid Price (Simple Microprice).
    pub fn weighted_mid_price(&self, depth_level: usize) -> Result<f64, MicropriceError> {
        let bid_len = self.bids.len();
        let ask_len = self.asks.len();

        if bid_len < depth_level || ask_len < depth_level {
            return Err(MicropriceError::InsufficientDepth {
                depth: depth_level,
                bid_len,
                ask_len,
            });
        }

        // Single pass calculation using fold for efficiency
        let (sum_bq, weighted_bid) = self
            .bids
            .iter()
            .take(depth_level)
            .fold((0.0, 0.0), |(q_acc, wp_acc), l| {
                (q_acc + l.quantity, wp_acc + l.price * l.quantity)
            });

        let (sum_aq, weighted_ask) = self
            .asks
            .iter()
            .take(depth_level)
            .fold((0.0, 0.0), |(q_acc, wp_acc), l| {
                (q_acc + l.quantity, wp_acc + l.price * l.quantity)
            });

        if sum_bq == 0.0 || sum_aq == 0.0 {
            return Err(MicropriceError::ZeroLiquidity);
        }

        let w_bid_px = weighted_bid / sum_bq;
        let w_ask_px = weighted_ask / sum_aq;

        // Microprice formula: weighted by OPPOSITE side liquidity
        Ok((w_bid_px * sum_aq + w_ask_px * sum_bq) / (sum_bq + sum_aq))
    }
}

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

// ... (existing imports)

/// Model for Stoikov Microprice Adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SStoikovMicroPrice {
    /// Upper bounds for each bucket (quantiles).
    /// If num_buckets is K, this vector has K-1 elements.
    pub bounds: Vec<f64>,
    /// Adjustment value (E[dP]) for each bucket. Has K elements.
    pub adjustments: Vec<f64>,
}

impl Default for SStoikovMicroPrice {
    fn default() -> Self {
        Self::new()
    }
}

impl SStoikovMicroPrice {
    /// Creates a new, empty model.
    pub fn new() -> Self {
        Self {
            bounds: Vec::new(),
            adjustments: Vec::new(),
        }
    }

    /// Train the model on historical data.
    /// `samples`: List of (imbalance, future_mid_change)
    /// `num_buckets`: Number of quantile buckets to use
    pub fn train(&mut self, samples: &[(f64, f64)], num_buckets: usize) {
        if samples.is_empty() || num_buckets == 0 {
            return;
        }

        let n = samples.len();
        // Fix: cap actual_buckets to n to avoid empty bucket error (Gemini advice)
        let actual_buckets = std::cmp::min(num_buckets, n);

        // Sort samples by imbalance to determine quantiles
        let mut sorted_samples = samples.to_vec();
        sorted_samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut bounds = Vec::new();
        let mut adjustments = Vec::new();

        // Calculate bucket boundaries and averages
        for i in 0..actual_buckets {
            // Start index for this bucket
            let start_idx = (i * n) / actual_buckets;
            // End index (exclusive)
            let end_idx = ((i + 1) * n) / actual_buckets;

            if start_idx >= end_idx {
                continue; // Should no longer happen with capped actual_buckets
            }

            // Calculate average return for this bucket
            let sum_ret: f64 = sorted_samples[start_idx..end_idx].iter().map(|s| s.1).sum();
            let avg_ret = sum_ret / (end_idx - start_idx) as f64;
            adjustments.push(avg_ret);

            // Store upper bound for this bucket (except for the last one)
            if i < actual_buckets - 1 {
                // The bound is the imbalance of the last element in this bucket
                let threshold = sorted_samples[end_idx - 1].0;
                bounds.push(threshold);
            }
        }

        self.bounds = bounds;
        self.adjustments = adjustments;
    }

    /// Get the adjustment for a given imbalance
    pub fn get_adjustment(&self, imbalance: f64) -> f64 {
        if self.adjustments.is_empty() {
            return 0.0;
        }

        // Find which bucket this imbalance falls into
        for (i, bound) in self.bounds.iter().enumerate() {
            if imbalance <= *bound {
                return self.adjustments[i];
            }
        }

        // If greater than all bounds, it's in the last bucket
        *self.adjustments.last().unwrap_or(&0.0)
    }

    /// Save the model to a JSON file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Load the model from a JSON file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model = serde_json::from_reader(reader)?;
        Ok(model)
    }
}

/// 2D Stoikov Microprice Model (Imbalance + Spread)
/// State = (spread_ticks, imbalance_bucket)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SStoikov2DMicroPrice {
    /// Upper bounds for imbalance buckets (K-1 elements for K buckets).
    pub imbalance_bounds: Vec<f64>,
    /// Minimum spread in ticks to track.
    pub min_spread_ticks: u32,
    /// Maximum spread in ticks to track. Spreads larger than this are capped.
    pub max_spread_ticks: u32,
    /// 2D Adjustment Matrix: [spread_ticks - min_spread_ticks][imbalance_bucket]
    pub adjustments: Vec<Vec<f64>>,
}

impl SStoikov2DMicroPrice {
    pub fn new(min_spread: u32, max_spread: u32) -> Self {
        let num_spreads = (max_spread - min_spread + 1) as usize;
        Self {
            imbalance_bounds: Vec::new(),
            min_spread_ticks: min_spread,
            max_spread_ticks: max_spread,
            adjustments: vec![Vec::new(); num_spreads],
        }
    }

    /// Returns the maximum spread observed in a sample set.
    pub fn measure_max_spread(samples: &[(f64, u32, f64)]) -> u32 {
        samples.iter().map(|s| s.1).max().unwrap_or(0)
    }

    /// Train the 2D model.
    /// `samples`: List of (imbalance, spread_ticks, future_mid_change)
    /// `num_buckets`: Number of imbalance buckets per spread level.
    pub fn train(&mut self, samples: &[(f64, u32, f64)], num_buckets: usize) {
        if samples.is_empty() || num_buckets == 0 {
            return;
        }

        // 1. Determine Imbalance Quantiles across ALL samples
        let mut sorted_imb: Vec<f64> = samples.iter().map(|s| s.0).collect();
        sorted_imb.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_imb.len();
        let actual_buckets = std::cmp::min(num_buckets, n);
        let mut imbalance_bounds = Vec::new();

        for i in 1..actual_buckets {
            let idx = (i * n) / actual_buckets;
            imbalance_bounds.push(sorted_imb[idx - 1]);
        }
        self.imbalance_bounds = imbalance_bounds;

        // 2. Accumulate sums and counts in a single pass O(N)
        let num_spreads = (self.max_spread_ticks - self.min_spread_ticks + 1) as usize;
        let mut sums = vec![vec![0.0; actual_buckets]; num_spreads];
        let mut counts = vec![vec![0usize; actual_buckets]; num_spreads];

        for &(imb, spread, change) in samples {
            let imb_idx = self.get_bucket_index(imb);
            let s_idx = (spread.clamp(self.min_spread_ticks, self.max_spread_ticks)
                - self.min_spread_ticks) as usize;

            if s_idx < num_spreads && imb_idx < actual_buckets {
                sums[s_idx][imb_idx] += change;
                counts[s_idx][imb_idx] += 1;
            }
        }

        // 3. Compute averages for adjustments with fallback
        self.adjustments = vec![vec![0.0; actual_buckets]; num_spreads];
        for s in 0..num_spreads {
            // Calculate the average for this specific spread level
            let row_sum: f64 = sums[s].iter().sum();
            let row_count: usize = counts[s].iter().sum();
            let row_avg = if row_count > 0 {
                row_sum / row_count as f64
            } else {
                0.0
            };

            for b in 0..actual_buckets {
                if counts[s][b] > 0 {
                    self.adjustments[s][b] = sums[s][b] / counts[s][b] as f64;
                } else {
                    // Fallback to the spread average instead of 0.0
                    self.adjustments[s][b] = row_avg;
                }
            }
        }
    }

    /// Get the adjustment for a given imbalance and spread
    pub fn get_adjustment(&self, imbalance: f64, spread_ticks: u32) -> f64 {
        let imb_idx = self.get_bucket_index(imbalance);
        let spread_idx = (spread_ticks.clamp(self.min_spread_ticks, self.max_spread_ticks)
            - self.min_spread_ticks) as usize;

        if spread_idx < self.adjustments.len() {
            let row = &self.adjustments[spread_idx];
            if imb_idx < row.len() {
                return row[imb_idx];
            }
        }
        0.0
    }

    fn get_bucket_index(&self, imbalance: f64) -> usize {
        for (i, bound) in self.imbalance_bounds.iter().enumerate() {
            if imbalance <= *bound {
                return i;
            }
        }
        self.imbalance_bounds.len()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mid_price() {
        let state = OrderBookState {
            bids: vec![Level {
                price: 100.0,
                quantity: 10.0,
            }],
            asks: vec![Level {
                price: 101.0,
                quantity: 20.0,
            }],
        };
        assert_eq!(state.mid_price().unwrap(), 100.5);
    }

    #[test]
    fn test_sstoikov_2d_train() {
        let mut model = SStoikov2DMicroPrice::new(1, 5);
        // (imb, spread, ret)
        let samples = vec![(0.1, 1, 1.0), (0.1, 1, 2.0), (0.9, 1, -1.0), (0.5, 2, 5.0)];
        model.train(&samples, 2); // 2 imbalance buckets

        // Spread 1, Imb 0.1 -> Bucket 0 -> Avg(1, 2) = 1.5
        assert_eq!(model.get_adjustment(0.1, 1), 1.5);
        // Spread 2, Imb 0.5 -> Bucket 0 or 1 depending on threshold?
        // With samples [0.1, 0.1, 0.5, 0.9], threshold for 2 buckets is sorted[n/2-1] = sorted[1] = 0.1
        // 0.5 > 0.1 => Bucket 1.
        assert_eq!(model.get_adjustment(0.5, 2), 5.0);
    }

    #[test]
    fn test_insufficient_depth_error() {
        let state = OrderBookState {
            bids: vec![Level {
                price: 100.0,
                quantity: 10.0,
            }],
            asks: vec![Level {
                price: 101.0,
                quantity: 20.0,
            }],
        };
        // Request depth 2 on a book with depth 1
        assert!(matches!(
            state.imbalance(2),
            Err(MicropriceError::InsufficientDepth { .. })
        ));
        assert!(matches!(
            state.weighted_mid_price(2),
            Err(MicropriceError::InsufficientDepth { .. })
        ));
    }
}
