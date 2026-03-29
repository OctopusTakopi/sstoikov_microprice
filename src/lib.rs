use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
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
    #[error("Requested depth must be greater than zero")]
    InvalidDepth,
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
        if depth_level == 0 {
            return Err(MicropriceError::InvalidDepth);
        }

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
        if depth_level == 0 {
            return Err(MicropriceError::InvalidDepth);
        }

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

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("cannot fit a model with zero samples")]
    EmptySamples,
    #[error("bucket count must be greater than zero")]
    InvalidBucketCount,
    #[error("sample {index} has a non-finite imbalance value: {value}")]
    NonFiniteImbalance { index: usize, value: f64 },
    #[error("sample {index} has a non-finite target value: {value}")]
    NonFiniteTarget { index: usize, value: f64 },
    #[error(
        "invalid 1D model shape: expected adjustments.len() == bounds.len() + 1, got bounds={bounds_len}, adjustments={adjustments_len}"
    )]
    InvalidBucketShape {
        bounds_len: usize,
        adjustments_len: usize,
    },
    #[error("model bound at index {index} is not finite: {value}")]
    NonFiniteBound { index: usize, value: f64 },
    #[error(
        "model bounds must be strictly increasing, but bounds[{previous_index}]={previous_value} and bounds[{index}]={value}"
    )]
    BoundsNotStrictlyIncreasing {
        previous_index: usize,
        previous_value: f64,
        index: usize,
        value: f64,
    },
    #[error("adjustment at index {index} is not finite: {value}")]
    NonFiniteAdjustment { index: usize, value: f64 },
    #[error(
        "spread range is invalid: min_spread_ticks={min_spread_ticks}, max_spread_ticks={max_spread_ticks}"
    )]
    InvalidSpreadRange {
        min_spread_ticks: u32,
        max_spread_ticks: u32,
    },
    #[error(
        "invalid 2D model row count: expected {expected} rows for the configured spread range, got {actual}"
    )]
    InvalidSpreadRowCount { expected: usize, actual: usize },
    #[error("row {row} has {actual} buckets, expected {expected}")]
    InvalidSpreadRowShape {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error("adjustment at row {row}, bucket {bucket} is not finite: {value}")]
    NonFinite2DAdjustment {
        row: usize,
        bucket: usize,
        value: f64,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
struct QuantileLayout {
    bounds: Vec<f64>,
    sample_ends: Vec<usize>,
}

impl QuantileLayout {
    fn bucket_count(&self) -> usize {
        self.sample_ends.len()
    }
}

fn validate_bucket_count(num_buckets: usize) -> Result<(), ModelError> {
    if num_buckets == 0 {
        Err(ModelError::InvalidBucketCount)
    } else {
        Ok(())
    }
}

fn validate_1d_samples(samples: &[(f64, f64)]) -> Result<(), ModelError> {
    if samples.is_empty() {
        return Err(ModelError::EmptySamples);
    }

    for (index, &(imbalance, target)) in samples.iter().enumerate() {
        if !imbalance.is_finite() {
            return Err(ModelError::NonFiniteImbalance {
                index,
                value: imbalance,
            });
        }
        if !target.is_finite() {
            return Err(ModelError::NonFiniteTarget {
                index,
                value: target,
            });
        }
    }

    Ok(())
}

fn validate_2d_samples(samples: &[(f64, u32, f64)]) -> Result<(), ModelError> {
    if samples.is_empty() {
        return Err(ModelError::EmptySamples);
    }

    for (index, &(imbalance, _spread_ticks, target)) in samples.iter().enumerate() {
        if !imbalance.is_finite() {
            return Err(ModelError::NonFiniteImbalance {
                index,
                value: imbalance,
            });
        }
        if !target.is_finite() {
            return Err(ModelError::NonFiniteTarget {
                index,
                value: target,
            });
        }
    }

    Ok(())
}

fn validate_bounds(bounds: &[f64]) -> Result<(), ModelError> {
    for (index, &bound) in bounds.iter().enumerate() {
        if !bound.is_finite() {
            return Err(ModelError::NonFiniteBound {
                index,
                value: bound,
            });
        }
        if index > 0 && bounds[index - 1] >= bound {
            return Err(ModelError::BoundsNotStrictlyIncreasing {
                previous_index: index - 1,
                previous_value: bounds[index - 1],
                index,
                value: bound,
            });
        }
    }

    Ok(())
}

fn build_quantile_layout(sorted_imbalances: &[f64], requested_buckets: usize) -> QuantileLayout {
    debug_assert!(!sorted_imbalances.is_empty());
    debug_assert!(requested_buckets > 0);

    let mut run_ends = Vec::with_capacity(sorted_imbalances.len());
    for index in 1..sorted_imbalances.len() {
        if sorted_imbalances[index - 1] != sorted_imbalances[index] {
            run_ends.push(index);
        }
    }
    run_ends.push(sorted_imbalances.len());

    let total_count = sorted_imbalances.len();
    let total_groups = run_ends.len();
    let actual_buckets = requested_buckets.min(total_groups);

    let mut bounds = Vec::with_capacity(actual_buckets.saturating_sub(1));
    let mut sample_ends = Vec::with_capacity(actual_buckets);
    let mut start_group = 0usize;

    for bucket_idx in 0..actual_buckets {
        let end_group = if bucket_idx + 1 == actual_buckets {
            total_groups
        } else {
            let groups_left_after = actual_buckets - bucket_idx - 1;
            let min_end = start_group + 1;
            let max_end = total_groups - groups_left_after;
            let target_scaled = (bucket_idx as u128 + 1) * total_count as u128;

            let mut candidate_end = min_end;
            while candidate_end < max_end
                && (run_ends[candidate_end - 1] as u128) * (actual_buckets as u128) < target_scaled
            {
                candidate_end += 1;
            }

            if candidate_end > min_end {
                let prev_count = run_ends[candidate_end - 2] as u128 * actual_buckets as u128;
                let next_count = run_ends[candidate_end - 1] as u128 * actual_buckets as u128;
                if target_scaled.abs_diff(prev_count) <= target_scaled.abs_diff(next_count) {
                    candidate_end -= 1;
                }
            }

            candidate_end
        };

        let sample_end = run_ends[end_group - 1];
        sample_ends.push(sample_end);
        if bucket_idx + 1 < actual_buckets {
            bounds.push(sorted_imbalances[sample_end - 1]);
        }
        start_group = end_group;
    }

    QuantileLayout {
        bounds,
        sample_ends,
    }
}

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

    /// Fit the model on historical data.
    /// `samples`: List of (imbalance, future_target_change)
    /// `num_buckets`: Maximum number of quantile buckets to use.
    ///
    /// Buckets are tie-aware: identical imbalance values are never split across
    /// two buckets, so the fitted thresholds match prediction-time lookup.
    pub fn fit(&mut self, samples: &[(f64, f64)], num_buckets: usize) -> Result<(), ModelError> {
        validate_bucket_count(num_buckets)?;
        validate_1d_samples(samples)?;

        let mut sorted_samples = samples.to_vec();
        sorted_samples.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        let sorted_imbalances: Vec<f64> = sorted_samples.iter().map(|sample| sample.0).collect();
        let layout = build_quantile_layout(&sorted_imbalances, num_buckets);

        let mut adjustments = Vec::with_capacity(layout.bucket_count());
        let mut start_idx = 0usize;
        for &end_idx in &layout.sample_ends {
            let sum_target: f64 = sorted_samples[start_idx..end_idx]
                .iter()
                .map(|sample| sample.1)
                .sum();
            adjustments.push(sum_target / (end_idx - start_idx) as f64);
            start_idx = end_idx;
        }

        self.bounds = layout.bounds;
        self.adjustments = adjustments;
        self.validate()
    }

    /// Compatibility wrapper for older callers. Prefer [`Self::fit`].
    pub fn train(&mut self, samples: &[(f64, f64)], num_buckets: usize) {
        let _ = self.fit(samples, num_buckets);
    }

    /// Get the adjustment for a given imbalance
    pub fn get_adjustment(&self, imbalance: f64) -> f64 {
        if self.adjustments.is_empty() || !imbalance.is_finite() {
            return 0.0;
        }

        // Binary search: find first bucket whose upper bound >= imbalance
        let idx = self.bounds.partition_point(|&b| b < imbalance);
        self.adjustments[idx.min(self.adjustments.len() - 1)]
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.adjustments.is_empty() {
            if self.bounds.is_empty() {
                return Ok(());
            }
            return Err(ModelError::InvalidBucketShape {
                bounds_len: self.bounds.len(),
                adjustments_len: self.adjustments.len(),
            });
        }

        if self.bounds.len() + 1 != self.adjustments.len() {
            return Err(ModelError::InvalidBucketShape {
                bounds_len: self.bounds.len(),
                adjustments_len: self.adjustments.len(),
            });
        }

        validate_bounds(&self.bounds)?;

        for (index, &adjustment) in self.adjustments.iter().enumerate() {
            if !adjustment.is_finite() {
                return Err(ModelError::NonFiniteAdjustment {
                    index,
                    value: adjustment,
                });
            }
        }

        Ok(())
    }

    /// Save the model to a JSON file.
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), ModelError> {
        self.validate()?;
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Load and validate the model from a JSON file.
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Self, ModelError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model: Self = serde_json::from_reader(reader)?;
        model.validate()?;
        Ok(model)
    }

    /// Compatibility wrapper for older callers. Prefer [`Self::save_model`].
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ModelError> {
        self.save_model(path)
    }

    /// Compatibility wrapper for older callers. Prefer [`Self::load_model`].
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ModelError> {
        Self::load_model(path)
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
        assert!(
            max_spread >= min_spread,
            "max_spread ({max_spread}) must be >= min_spread ({min_spread})"
        );
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

    /// Fit the 2D model.
    /// `samples`: List of (imbalance, spread_ticks, future_target_change)
    /// `num_buckets`: Maximum number of imbalance buckets per spread level.
    pub fn fit(
        &mut self,
        samples: &[(f64, u32, f64)],
        num_buckets: usize,
    ) -> Result<(), ModelError> {
        validate_bucket_count(num_buckets)?;
        validate_2d_samples(samples)?;

        let mut sorted_imbalance = samples.iter().map(|sample| sample.0).collect::<Vec<_>>();
        sorted_imbalance.sort_unstable_by(f64::total_cmp);
        let layout = build_quantile_layout(&sorted_imbalance, num_buckets);
        let bucket_count = layout.bucket_count();
        self.imbalance_bounds = layout.bounds;

        let num_spreads = (self.max_spread_ticks - self.min_spread_ticks + 1) as usize;
        let mut sums = vec![vec![0.0; bucket_count]; num_spreads];
        let mut counts = vec![vec![0usize; bucket_count]; num_spreads];

        for &(imbalance, spread_ticks, target) in samples {
            let bucket_idx = self.get_bucket_index(imbalance);
            let spread_idx = (spread_ticks.clamp(self.min_spread_ticks, self.max_spread_ticks)
                - self.min_spread_ticks) as usize;
            sums[spread_idx][bucket_idx] += target;
            counts[spread_idx][bucket_idx] += 1;
        }

        self.adjustments = vec![vec![0.0; bucket_count]; num_spreads];
        for spread_idx in 0..num_spreads {
            let row_sum: f64 = sums[spread_idx].iter().sum();
            let row_count: usize = counts[spread_idx].iter().sum();
            let row_average = if row_count > 0 {
                row_sum / row_count as f64
            } else {
                0.0
            };

            for bucket_idx in 0..bucket_count {
                self.adjustments[spread_idx][bucket_idx] = if counts[spread_idx][bucket_idx] > 0 {
                    sums[spread_idx][bucket_idx] / counts[spread_idx][bucket_idx] as f64
                } else {
                    row_average
                };
            }
        }

        self.validate()
    }

    /// Compatibility wrapper for older callers. Prefer [`Self::fit`].
    pub fn train(&mut self, samples: &[(f64, u32, f64)], num_buckets: usize) {
        let _ = self.fit(samples, num_buckets);
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.max_spread_ticks < self.min_spread_ticks {
            return Err(ModelError::InvalidSpreadRange {
                min_spread_ticks: self.min_spread_ticks,
                max_spread_ticks: self.max_spread_ticks,
            });
        }

        validate_bounds(&self.imbalance_bounds)?;

        let expected_rows = (self.max_spread_ticks - self.min_spread_ticks + 1) as usize;
        if self.adjustments.len() != expected_rows {
            return Err(ModelError::InvalidSpreadRowCount {
                expected: expected_rows,
                actual: self.adjustments.len(),
            });
        }

        let is_unfitted = self.adjustments.iter().all(Vec::is_empty);
        if is_unfitted {
            if self.imbalance_bounds.is_empty() {
                return Ok(());
            }
            return Err(ModelError::InvalidBucketShape {
                bounds_len: self.imbalance_bounds.len(),
                adjustments_len: 0,
            });
        }

        let expected_bucket_count = self.imbalance_bounds.len() + 1;
        for (row, buckets) in self.adjustments.iter().enumerate() {
            if buckets.len() != expected_bucket_count {
                return Err(ModelError::InvalidSpreadRowShape {
                    row,
                    expected: expected_bucket_count,
                    actual: buckets.len(),
                });
            }
            for (bucket, &value) in buckets.iter().enumerate() {
                if !value.is_finite() {
                    return Err(ModelError::NonFinite2DAdjustment { row, bucket, value });
                }
            }
        }

        Ok(())
    }

    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), ModelError> {
        self.validate()?;
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Self, ModelError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model: Self = serde_json::from_reader(reader)?;
        model.validate()?;
        Ok(model)
    }

    /// Get the adjustment for a given imbalance and spread
    pub fn get_adjustment(&self, imbalance: f64, spread_ticks: u32) -> f64 {
        if !imbalance.is_finite() {
            return 0.0;
        }

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
        // Binary search: first bound that is >= imbalance
        self.imbalance_bounds.partition_point(|&b| b < imbalance)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ModelError> {
        self.save_model(path)
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ModelError> {
        Self::load_model(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_model_path(prefix: &str) -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "sstoikov_microprice_{prefix}_{}_{}.json",
            std::process::id(),
            timestamp
        ))
    }

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
    fn test_sstoikov_fit_is_tie_aware() {
        let samples = vec![(0.1, 1.0), (0.1, 3.0), (0.5, 2.0), (0.9, -1.0), (0.9, -3.0)];

        let mut model = SStoikovMicroPrice::new();
        model.fit(&samples, 4).unwrap();

        assert_eq!(model.bounds, vec![0.1, 0.5]);
        assert_eq!(model.adjustments, vec![2.0, 2.0, -2.0]);
        assert_eq!(model.get_adjustment(0.1), 2.0);
        assert_eq!(model.get_adjustment(0.5), 2.0);
        assert_eq!(model.get_adjustment(0.9), -2.0);
    }

    #[test]
    fn test_sstoikov_fit_rejects_non_finite_samples() {
        let mut model = SStoikovMicroPrice::new();
        let error = model.fit(&[(f64::NAN, 1.0)], 2).unwrap_err();
        assert!(matches!(error, ModelError::NonFiniteImbalance { .. }));
    }

    #[test]
    fn test_sstoikov_predict_rejects_non_finite_imbalance() {
        let mut model = SStoikovMicroPrice::new();
        model.fit(&[(0.1, 1.0), (0.9, -1.0)], 2).unwrap();
        assert_eq!(model.get_adjustment(f64::NAN), 0.0);
        assert_eq!(model.get_adjustment(f64::INFINITY), 0.0);
    }

    #[test]
    fn test_sstoikov_model_round_trip() {
        let path = temp_model_path("1d_round_trip");
        let samples = vec![(0.1, 1.0), (0.1, 3.0), (0.9, -1.0), (0.9, -3.0)];

        let mut model = SStoikovMicroPrice::new();
        model.fit(&samples, 4).unwrap();
        model.save_model(&path).unwrap();

        let loaded = SStoikovMicroPrice::load_model(&path).unwrap();
        assert_eq!(loaded.bounds, model.bounds);
        assert_eq!(loaded.adjustments, model.adjustments);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_sstoikov_load_rejects_invalid_shape() {
        let path = temp_model_path("1d_invalid");
        fs::write(&path, r#"{"bounds":[0.5],"adjustments":[]}"#).unwrap();

        let error = SStoikovMicroPrice::load_model(&path).unwrap_err();
        assert!(matches!(error, ModelError::InvalidBucketShape { .. }));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_sstoikov_2d_fit() {
        let mut model = SStoikov2DMicroPrice::new(1, 5);
        // (imb, spread, ret)
        let samples = vec![(0.1, 1, 1.0), (0.1, 1, 2.0), (0.9, 1, -1.0), (0.5, 2, 5.0)];
        model.fit(&samples, 2).unwrap(); // 2 imbalance buckets

        // Spread 1, Imb 0.1 -> Bucket 0 -> Avg(1, 2) = 1.5
        assert_eq!(model.get_adjustment(0.1, 1), 1.5);
        // With tie-aware buckets, the split remains at 0.1, so 0.5 maps to bucket 1.
        assert_eq!(model.get_adjustment(0.5, 2), 5.0);
    }

    #[test]
    fn test_sstoikov_2d_model_round_trip() {
        let path = temp_model_path("2d_round_trip");
        let samples = vec![(0.1, 1, 1.0), (0.1, 1, 2.0), (0.9, 2, -1.0), (0.9, 2, -3.0)];

        let mut model = SStoikov2DMicroPrice::new(1, 2);
        model.fit(&samples, 4).unwrap();
        model.save_model(&path).unwrap();

        let loaded = SStoikov2DMicroPrice::load_model(&path).unwrap();
        assert_eq!(loaded.imbalance_bounds, model.imbalance_bounds);
        assert_eq!(loaded.adjustments, model.adjustments);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_sstoikov_2d_predict_rejects_non_finite_imbalance() {
        let mut model = SStoikov2DMicroPrice::new(1, 2);
        model.fit(&[(0.1, 1, 1.0), (0.9, 2, -1.0)], 2).unwrap();
        assert_eq!(model.get_adjustment(f64::NAN, 1), 0.0);
        assert_eq!(model.get_adjustment(f64::NEG_INFINITY, 2), 0.0);
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

    #[test]
    fn test_invalid_depth_error() {
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

        assert!(matches!(
            state.imbalance(0),
            Err(MicropriceError::InvalidDepth)
        ));
        assert!(matches!(
            state.weighted_mid_price(0),
            Err(MicropriceError::InvalidDepth)
        ));
    }
}
