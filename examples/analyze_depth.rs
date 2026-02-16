use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[path = "local_orderbook/mod.rs"]
mod local_orderbook;

use crate::local_orderbook::{DepthUpdateEvent, LocalOrderbook};
use sstoikov_microprice::SStoikovMicroPrice;

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "data/btcusdt_20260210.gz";
    println!("Loading depth data from {}", file_path);

    let file = File::open(file_path)?;
    let decoder = GzDecoder::new(file);
    let reader = BufReader::new(decoder);

    // Reconstruct Orderbook
    // Assuming 0.01 tick size for BTCUSDT
    let mut ob = LocalOrderbook::new(0.01);

    // Store features for analysis
    // (Timestamp, MidPrice, Map<DepthKey, (SimpleMP, Imbalance)>)
    struct Row {
        ts: u64,
        mid: f64,
        metrics: HashMap<usize, (f64, f64)>, // Depth -> (SimpleMP, Imbalance)
    }

    let mut history: Vec<Row> = Vec::new();
    let depths = [2, 5, 10, 20, 50, 100];

    let mut event_count = 0;

    for line_res in reader.lines() {
        let line = line_res?;
        let Some((ts_str, json_str)) = line.split_once(' ') else {
            continue;
        };
        let Ok(ts_nanos) = ts_str.parse::<u64>() else {
            continue;
        };
        let ts_millis = ts_nanos / 1_000_000;

        if json_str.contains("depthUpdate") {
            match serde_json::from_str::<DepthUpdateEvent>(json_str) {
                Ok(event) => {
                    ob.process_diff_book_depth(event.data);

                    let Some(state) = ob.get_state(100) else {
                        continue;
                    };
                    let Ok(mid) = state.mid_price() else {
                        continue;
                    };
                    let mut metrics = HashMap::new();
                    for &d in &depths {
                        let simple_mp = state.weighted_mid_price(d).unwrap_or(mid);
                        let imb = state.imbalance(d).unwrap_or(0.5);
                        metrics.insert(d, (simple_mp, imb));
                    }

                    history.push(Row {
                        ts: ts_millis,
                        mid,
                        metrics,
                    });
                    event_count += 1;
                }
                Err(e) => {
                    if event_count == 0 {
                        println!("Failed to parse line: {}", json_str);
                        println!("Error: {}", e);
                    }
                }
            }
        }
    }

    println!(
        "Processed {} depth updates. History size: {}",
        event_count,
        history.len()
    );

    if history.is_empty() {
        return Ok(());
    }

    // --- IC Analysis ---
    println!("Computing ICs for various depths...");

    let horizons = [200, 500, 1000, 5000];

    for &horizon_ms in &horizons {
        println!("\nComputing ICs for {}ms horizon...", horizon_ms);
        let mut targets = Vec::new();
        let mut valid_indices = Vec::new(); // Indices in history that have a valid target

        // Compute Targets
        for (i, row) in history.iter().enumerate() {
            let target_ts = row.ts + horizon_ms;
            // Binary search for future price
            let next_idx = match history.binary_search_by_key(&target_ts, |r| r.ts) {
                Ok(idx) => idx,
                Err(idx) => idx,
            };

            if next_idx < history.len() {
                let next_row = &history[next_idx];
                let dt = next_row.ts as i64 - row.ts as i64;
                if (dt - horizon_ms as i64).abs() < 100 {
                    let ret_bps = (next_row.mid / row.mid - 1.0) * 10000.0;
                    targets.push(ret_bps);
                    valid_indices.push(i);
                }
            }
        }

        println!("Valid samples: {}", targets.len());

        if targets.len() < 100 {
            println!("Not enough samples for this horizon.");
            continue;
        }

        println!("{:=<80}", "");
        println!(
            "{:<6} | {:<15} | {:<15} | {:<15}",
            "Depth", "Simple IC", "Imbalance IC", "Stoikov Imbalance IC"
        );
        println!("{:=<80}", "");

        // SPLIT DATA: 50% Train, 50% Test
        let split_idx = valid_indices.len() / 2;
        let (train_indices, test_indices) = valid_indices.split_at(split_idx);

        for &d in &depths {
            // Calibration Data for Stoikov (TRAIN SET)
            let mut samples_for_train = Vec::new();

            for (idx, &hist_idx) in train_indices.iter().enumerate() {
                let row = &history[hist_idx];

                if let Some(&(_s_mp, imb)) = row.metrics.get(&d) {
                    let change = row.mid * targets[idx] / 10000.0;
                    samples_for_train.push((imb, change));
                } else {
                    samples_for_train.push((0.5, 0.0));
                }
            }

            // Train Stoikov
            let mut model = SStoikovMicroPrice::new();
            model.train(&samples_for_train, 10);

            // EVALUATION (TEST SET)
            let mut simple_preds = Vec::new();
            let mut imb_preds = Vec::new();
            let mut stoikov_preds = Vec::new();

            let test_targets = &targets[split_idx..];

            for &hist_idx in test_indices.iter() {
                let row = &history[hist_idx];
                if let Some(&(s_mp, imb)) = row.metrics.get(&d) {
                    simple_preds.push(s_mp - row.mid);
                    imb_preds.push(imb);
                    stoikov_preds.push(model.get_adjustment(imb));
                } else {
                    simple_preds.push(0.0);
                    imb_preds.push(0.5);
                    stoikov_preds.push(0.0);
                }
            }
            // Calc IC
            if simple_preds.len() == test_targets.len() {
                let ic_simple = pearson_correlation(&simple_preds, test_targets);
                let ic_imb = pearson_correlation(&imb_preds, test_targets);
                let ic_stoikov = pearson_correlation(&stoikov_preds, test_targets);

                println!(
                    "L{:<5} | {:>15.4} | {:>15.4} | {:>15.4}",
                    d, ic_simple, ic_imb, ic_stoikov
                );
            }
        }
        println!("{:=<80}", "");
    }

    Ok(())
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(xi, yi)| xi * yi).sum();
    let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_yy: f64 = y.iter().map(|yi| yi * yi).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denom = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();

    if denom == 0.0 { 0.0 } else { numerator / denom }
}
