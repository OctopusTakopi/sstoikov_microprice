use flate2::read::GzDecoder;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer};
use sstoikov_microprice::{Level, OrderBookState, SStoikovMicroPrice};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};

type SymbolStr = String;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BinanceSpotBookTickerRoot {
    pub stream: String,
    pub data: BinanceSpotBookTickerEvent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BinanceSpotBookTickerEvent {
    #[serde(rename = "u")]
    pub update_id: u64,
    #[serde(rename = "s")]
    pub symbol: SymbolStr,
    #[serde(rename = "b", deserialize_with = "from_str_to_f64")]
    pub best_bid: f64,
    #[serde(rename = "B", deserialize_with = "from_str_to_f64")]
    pub best_bid_qty: f64,
    #[serde(rename = "a", deserialize_with = "from_str_to_f64")]
    pub best_ask: f64,
    #[serde(rename = "A", deserialize_with = "from_str_to_f64")]
    pub best_ask_qty: f64,
}

struct F64Visitor;

impl<'de> Visitor<'de> for F64Visitor {
    type Value = Option<f64>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a string containing an f64 number")
    }

    fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if s.is_empty() {
            Ok(None)
        } else {
            Ok(Some(s.parse::<f64>().map_err(E::custom)?))
        }
    }
}

pub fn from_str_to_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer
        .deserialize_str(F64Visitor)
        .map(|value| value.unwrap_or(0.0))
}

struct Snapshot {
    ts: u64, // millis
    mid: f64,
    simple_mp: f64,
    imbalance: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "data/btcusdt_20260210.gz";
    println!("Loading data from {}", file_path);

    let file = File::open(file_path)?;
    let decoder = GzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut snapshots = Vec::new();

    for line_res in reader.lines() {
        let line = line_res?;
        // Format: "ts json_string"
        let Some((ts_str, json_str)) = line.split_once(' ') else {
            continue;
        };
        let Ok(ts_nanos) = ts_str.parse::<u64>() else {
            continue;
        };
        let ts_millis = ts_nanos / 1_000_000;

        // Parse JSON
        // We only care about BookTickers
        if json_str.contains("bookTicker") {
            let event: BinanceSpotBookTickerRoot = serde_json::from_str(json_str)?;

            let bid_px = event.data.best_bid;
            let bid_qty = event.data.best_bid_qty;
            let ask_px = event.data.best_ask;
            let ask_qty = event.data.best_ask_qty;

            if bid_px == 0.0 || ask_px == 0.0 {
                continue;
            }

            let state = OrderBookState {
                bids: vec![Level {
                    price: bid_px,
                    quantity: bid_qty,
                }],
                asks: vec![Level {
                    price: ask_px,
                    quantity: ask_qty,
                }],
            };

            if let Ok(mid) = state.mid_price() {
                let simple_mp = state.weighted_mid_price(1).unwrap_or(mid);
                let imbalance = state.imbalance(1).unwrap_or(0.5);

                snapshots.push(Snapshot {
                    ts: ts_millis,
                    mid,
                    simple_mp,
                    imbalance,
                });
            }
        }
    }

    println!("Loaded {} snapshots.", snapshots.len());
    if snapshots.is_empty() {
        return Ok(());
    }

    // Sort by timestamp just in case
    snapshots.sort_by_key(|s| s.ts);

    let start_time = snapshots.first().unwrap().ts;
    let end_time = snapshots.last().unwrap().ts;
    println!(
        "Time range: {} -> {} ({} ms)",
        start_time,
        end_time,
        end_time - start_time
    );

    // --- Analysis ---

    // Horizons in ms
    let horizons = [500, 1000, 5000];

    // Evaluation data
    // Metric -> Vec<(Prediction, TargetReturnBps)>
    // We want to evaluate IC for:
    // 1. Simple MP Signal (MP - Mid) vs Future Return
    // 2. Stoikov Signal (Adjustment) vs Future Return (Using 500ms model for all? Or train specific models?)
    // The user asked for "test the IC of future 500ms return and future 1s return and future 5s return"
    // And "compute top1 book's microprice matrix and show the matrix"
    // The matrix usually implies plotting Imbalance Buckets vs Future Returns.

    // Let's train a Stoikov model for EACH horizon to see the matrix for each,
    // or just train on 500ms as the "primary" microprice model and test its alpha on longer horizons.
    // Usually Microprice is for short term. Let's train on 500ms for the Matrix display.

    // Create datasets for calibration (500ms) and IC calc (all)
    // For efficient lookups of "future price", we can use binary search on timestamps since they are sorted.

    let mut samples_500ms = Vec::new();

    let mut preds_simple = vec![Vec::new(); horizons.len()];
    let mut preds_imbalance = vec![Vec::new(); horizons.len()];
    let mut targets = vec![Vec::new(); horizons.len()];
    let mut valid_indices = vec![Vec::new(); horizons.len()]; // To map back for Stoikov evaluation

    // Verify sort order
    if snapshots.windows(2).any(|w| w[0].ts > w[1].ts) {
        eprintln!("WARNING: Snapshots are NOT sorted by timestamp! Binary search will fail.");
    } else {
        println!("Verified: Snapshots are sorted by timestamp.");
    }

    println!("Computing future returns...");

    for (i, snap) in snapshots.iter().enumerate() {
        for (h_idx, &horizon_ms) in horizons.iter().enumerate() {
            let target_ts = snap.ts + horizon_ms;

            // Find snapshot closest to target_ts
            // binary search for raw index
            let next_idx = match snapshots.binary_search_by_key(&target_ts, |s| s.ts) {
                Ok(idx) => idx,
                Err(idx) => idx, // Insert position, close enough
            };

            if next_idx < snapshots.len() {
                let future_mid = snapshots[next_idx].mid;
                // Check if time diff is reasonable (within 10% of horizon?) or just take whatever is next?
                // Given HFT data, gaps might exist.
                let dt = snapshots[next_idx].ts as i64 - snap.ts as i64;
                if (dt - horizon_ms as i64).abs() < 100 {
                    // Allow 100ms tolerance
                    let ret_bps = (future_mid / snap.mid - 1.0) * 10000.0;
                    let mid_change = future_mid - snap.mid;

                    if horizon_ms == 500 {
                        samples_500ms.push((snap.imbalance, mid_change));
                    }

                    preds_simple[h_idx].push(snap.simple_mp - snap.mid);
                    preds_imbalance[h_idx].push(snap.imbalance);
                    targets[h_idx].push(ret_bps);
                    valid_indices[h_idx].push(i);
                }
            }
        }
    }

    println!("Finished processing snapshots.");
    for (h_idx, &h_ms) in horizons.iter().enumerate() {
        println!(
            "Horizon {}ms: {} valid samples.",
            h_ms,
            targets[h_idx].len()
        );
    }

    // Train Stoikov Model (on 500ms data)
    let num_buckets = 10;
    println!(
        "Training Stoikov Model (k={}) on 500ms horizon...",
        num_buckets
    );
    // SPLIT DATA: 50% Train, 50% Test
    let split_idx = samples_500ms.len() / 2;
    let (train_samples, _test_samples) = samples_500ms.split_at(split_idx);

    // Train Stoikov Model on TRAIN SET
    let mut stoikov_model = SStoikovMicroPrice::new();
    stoikov_model.train(train_samples, num_buckets);

    // Save and Load Verification
    let model_path = "stoikov_bookticker.json";
    if let Err(e) = stoikov_model.save_to_file(model_path) {
        eprintln!("Failed to save model: {}", e);
    } else {
        println!("Model saved to {}", model_path);
        // Verify Load
        match SStoikovMicroPrice::load_from_file(model_path) {
            Ok(loaded_model) => {
                println!(
                    "Model loaded successfully. Params: {} adjustments",
                    loaded_model.adjustments.len()
                );
                // Optional: Assert equality of adjustments
            }
            Err(e) => eprintln!("Failed to load model: {}", e),
        }
    }

    // Print Matrix
    println!("{:-<60}", "");
    println!(
        "{:<10} | {:>15} | {:>15} | {:>15}",
        "Horizon", "Simple IC", "Imbalance IC", "Stoikov Imbalance IC"
    );
    println!("{:-<60}", "");

    for (h_idx, &h_ms) in horizons.iter().enumerate() {
        let preds_s = &preds_simple[h_idx];
        let preds_imb = &preds_imbalance[h_idx];
        let targs = &targets[h_idx];
        let v_indices = &valid_indices[h_idx];

        let n = preds_s.len();
        if n < 100 {
            continue;
        }

        // Split this horizon's data 50/50
        let split = n / 2;

        let test_preds_s = &preds_s[split..];
        let test_preds_imb = &preds_imb[split..];
        let test_targs = &targs[split..];
        let test_v_indices = &v_indices[split..];

        // Compute Stoikov Test Preds
        let mut test_preds_stoikov = Vec::with_capacity(test_v_indices.len());
        for &idx in test_v_indices {
            let imb = snapshots[idx].imbalance;
            test_preds_stoikov.push(stoikov_model.get_adjustment(imb));
        }

        let ic_simple = pearson_correlation(test_preds_s, test_targs);
        let ic_stoikov = pearson_correlation(&test_preds_stoikov, test_targs); // Stoikov vs Return

        // Also Imbalance vs Return?
        // simple_mp = weighted_mid. weighted_mid - mid is the signal.
        // Imbalance is just 0..1.
        // Let's check correlation of Imbalance itself too.
        let ic_imb = pearson_correlation(test_preds_imb, test_targs);

        println!(
            "{:<10} | {:>15.4} | {:>15.4} | {:>15.4}",
            h_ms, ic_simple, ic_imb, ic_stoikov
        );
    }
    println!("{:=<60}", "");

    Ok(())
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut covariance = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        covariance += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x <= 0.0 || var_y <= 0.0 {
        0.0
    } else {
        covariance / (var_x.sqrt() * var_y.sqrt())
    }
}
