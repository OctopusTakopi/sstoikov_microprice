use std::env;
use std::error::Error;
use std::path::PathBuf;

use sstoikov_microprice::{
    BinanceEvent, RawSbeReader, SStoikov2DMicroPrice, SStoikovMicroPrice, StoikovMarkovConfig,
    StoikovMarkovMicroPrice, StoikovObservation,
};

const RAW_DATA_DIR: &str = "/home/zp/rust-project/hl_hft/data/sbe/binance/spot";
const MOVE_HORIZONS: [usize; 2] = [1, 6];
const TIME_HORIZONS_MS: [i64; 5] = [500, 1_000, 5_000, 10_000, 15_000];

#[derive(Debug, Clone)]
struct BenchmarkObservation {
    stoikov: StoikovObservation,
    weighted_mid_adjustment: f64,
}

#[derive(Debug, Clone)]
struct TimeTargetSet {
    targets: Vec<Option<f64>>,
    move_counts: Vec<Option<usize>>,
}

#[derive(Debug, Clone, Copy)]
struct EvalMetrics {
    samples: usize,
    weighted_ic: f64,
    legacy_1d_ic: f64,
    legacy_2d_ic: f64,
    stoikov_ic: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().collect::<Vec<_>>();
    let symbol = args.get(1).map(String::as_str).unwrap_or("btcusdt");
    let train_date = args.get(2).map(String::as_str).unwrap_or("20260304");
    let test_date = args.get(3).map(String::as_str).unwrap_or("20260305");
    let tick_size = args
        .get(4)
        .map(|value| value.parse::<f64>())
        .transpose()?
        .unwrap_or(0.01);

    let train_path = day_path(symbol, train_date);
    let test_path = day_path(symbol, test_date);

    println!("Loading SBE best-bid/ask observations...");
    let train_obs = load_bba_observations(&train_path, 0, tick_size)?;
    let test_obs = load_bba_observations(&test_path, 1, tick_size)?;

    println!(
        "Train observations: {} from {}",
        train_obs.len(),
        train_path.display()
    );
    println!(
        "Test observations : {} from {}",
        test_obs.len(),
        test_path.display()
    );

    if train_obs.len() < 2 || test_obs.len() < 2 {
        return Err("not enough BBA observations to benchmark".into());
    }

    let max_spread_ticks = choose_spread_cap(&train_obs);
    println!("Using spread state range: 1..={} ticks", max_spread_ticks);

    let stoikov_train = train_obs.iter().map(|obs| obs.stoikov).collect::<Vec<_>>();
    let mut stoikov_model = StoikovMarkovMicroPrice::new();
    let stoikov_config = StoikovMarkovConfig {
        num_imbalance_buckets: 10,
        min_spread_ticks: 1,
        max_spread_ticks,
        symmetrize: true,
        default_horizon_moves: 6,
    };
    stoikov_model.fit(&stoikov_train, tick_size, &stoikov_config)?;

    println!("{:-<110}", "");
    println!(
        "{:<16} | {:>8} | {:>12} | {:>12} | {:>12} | {:>12}",
        "Target (moves)", "Samples", "WeightedMid", "Legacy1D", "Legacy2D", "Stoikov"
    );
    println!("{:-<110}", "");

    for &horizon in &MOVE_HORIZONS {
        let train_targets = future_move_targets(&train_obs, horizon);
        let test_targets = future_move_targets(&test_obs, horizon);
        let metrics = evaluate_target(
            &train_obs,
            &train_targets,
            &test_obs,
            &test_targets,
            max_spread_ticks,
            &stoikov_model,
            horizon,
        )?;

        println!(
            "{:<16} | {:>8} | {:>12.4} | {:>12.4} | {:>12.4} | {:>12.4}",
            format!("tau_{}", horizon),
            metrics.samples,
            metrics.weighted_ic,
            metrics.legacy_1d_ic,
            metrics.legacy_2d_ic,
            metrics.stoikov_ic,
        );
    }

    println!("{:-<110}", "");
    println!("{:-<122}", "");
    println!(
        "{:<16} | {:>8} | {:>12} | {:>12} | {:>12} | {:>12} | {:>10}",
        "Target (time)", "Samples", "WeightedMid", "Legacy1D", "Legacy2D", "Stoikov", "StoikovTau"
    );
    println!("{:-<122}", "");

    for &horizon_ms in &TIME_HORIZONS_MS {
        let train_target_set = future_time_targets(&train_obs, horizon_ms * 1_000);
        let test_target_set = future_time_targets(&test_obs, horizon_ms * 1_000);
        let stoikov_move_horizon =
            estimate_equivalent_move_horizon(&train_obs, &train_target_set, max_spread_ticks);
        let metrics = evaluate_target(
            &train_obs,
            &train_target_set.targets,
            &test_obs,
            &test_target_set.targets,
            max_spread_ticks,
            &stoikov_model,
            stoikov_move_horizon,
        )?;

        println!(
            "{:<16} | {:>8} | {:>12.4} | {:>12.4} | {:>12.4} | {:>12.4} | {:>10}",
            format_time_label(horizon_ms),
            metrics.samples,
            metrics.weighted_ic,
            metrics.legacy_1d_ic,
            metrics.legacy_2d_ic,
            metrics.stoikov_ic,
            stoikov_move_horizon,
        );
    }

    println!("{:-<122}", "");
    Ok(())
}

fn day_path(symbol: &str, raw_date: &str) -> PathBuf {
    let normalized_date = raw_date.replace('-', "");
    PathBuf::from(RAW_DATA_DIR).join(format!("{}_{}.zst", symbol.to_lowercase(), normalized_date))
}

fn load_bba_observations(
    path: &PathBuf,
    sequence_id: u64,
    tick_size: f64,
) -> Result<Vec<BenchmarkObservation>, Box<dyn Error>> {
    let mut reader = RawSbeReader::new(vec![path.clone()]);
    let mut payload = Vec::new();
    let mut observations = Vec::new();

    while let Some(record) = reader.next_record(&mut payload)? {
        if record.tag != b'S' {
            continue;
        }
        let Some(BinanceEvent::BestBidAsk(event)) = BinanceEvent::from_sbe(&payload) else {
            continue;
        };

        if !event.best_bid.is_finite()
            || !event.best_ask.is_finite()
            || !event.best_bid_qty.is_finite()
            || !event.best_ask_qty.is_finite()
        {
            continue;
        }
        if event.best_bid <= 0.0 || event.best_ask <= 0.0 {
            continue;
        }

        let total_qty = event.best_bid_qty + event.best_ask_qty;
        if total_qty <= 0.0 {
            continue;
        }

        let mid_price = (event.best_bid + event.best_ask) / 2.0;
        let weighted_mid =
            (event.best_bid * event.best_ask_qty + event.best_ask * event.best_bid_qty) / total_qty;
        let spread_ticks = ((event.best_ask - event.best_bid) / tick_size).round() as i64;
        if spread_ticks <= 0 {
            continue;
        }

        observations.push(BenchmarkObservation {
            stoikov: StoikovObservation {
                sequence_id,
                event_time_us: event.event_time_us,
                mid_price,
                imbalance: event.best_bid_qty / total_qty,
                spread_ticks: spread_ticks as u32,
            },
            weighted_mid_adjustment: weighted_mid - mid_price,
        });
    }

    Ok(observations)
}

fn choose_spread_cap(observations: &[BenchmarkObservation]) -> u32 {
    let mut spreads = observations
        .iter()
        .map(|obs| obs.stoikov.spread_ticks)
        .filter(|&spread| spread > 0)
        .collect::<Vec<_>>();
    if spreads.is_empty() {
        return 1;
    }
    spreads.sort_unstable();
    let percentile_index = ((spreads.len() as f64) * 0.99).floor() as usize;
    spreads[percentile_index.min(spreads.len() - 1)].clamp(1, 8)
}

fn evaluate_target(
    train_obs: &[BenchmarkObservation],
    train_targets: &[Option<f64>],
    test_obs: &[BenchmarkObservation],
    test_targets: &[Option<f64>],
    max_spread_ticks: u32,
    stoikov_model: &StoikovMarkovMicroPrice,
    stoikov_move_horizon: usize,
) -> Result<EvalMetrics, Box<dyn Error>> {
    let mut legacy_1d_samples = Vec::new();
    let mut legacy_2d_samples = Vec::new();
    for (observation, target) in train_obs.iter().zip(train_targets.iter()) {
        let spread_ticks = observation.stoikov.spread_ticks;
        if spread_ticks > max_spread_ticks {
            continue;
        }
        if let Some(target) = target {
            legacy_1d_samples.push((observation.stoikov.imbalance, *target));
            legacy_2d_samples.push((observation.stoikov.imbalance, spread_ticks, *target));
        }
    }

    let mut legacy_1d = SStoikovMicroPrice::new();
    legacy_1d.fit(&legacy_1d_samples, 10)?;

    let mut legacy_2d = SStoikov2DMicroPrice::new(1, max_spread_ticks);
    legacy_2d.fit(&legacy_2d_samples, 10)?;

    let stoikov_adjustments = stoikov_model.adjustments_for_horizon(stoikov_move_horizon);
    let mut weighted_preds = Vec::new();
    let mut legacy_1d_preds = Vec::new();
    let mut legacy_2d_preds = Vec::new();
    let mut stoikov_preds = Vec::new();
    let mut realized = Vec::new();

    for (observation, target) in test_obs.iter().zip(test_targets.iter()) {
        let spread_ticks = observation.stoikov.spread_ticks;
        if spread_ticks > max_spread_ticks {
            continue;
        }
        let Some(target) = target else { continue };

        weighted_preds.push(observation.weighted_mid_adjustment);
        legacy_1d_preds.push(legacy_1d.get_adjustment(observation.stoikov.imbalance));
        legacy_2d_preds.push(legacy_2d.get_adjustment(
            observation.stoikov.imbalance,
            observation.stoikov.spread_ticks,
        ));
        let stoikov_idx = stoikov_model
            .state_index(
                observation.stoikov.imbalance,
                observation.stoikov.spread_ticks,
            )
            .unwrap_or(0);
        stoikov_preds.push(stoikov_adjustments[stoikov_idx]);
        realized.push(*target);
    }

    Ok(EvalMetrics {
        samples: realized.len(),
        weighted_ic: pearson_correlation(&weighted_preds, &realized),
        legacy_1d_ic: pearson_correlation(&legacy_1d_preds, &realized),
        legacy_2d_ic: pearson_correlation(&legacy_2d_preds, &realized),
        stoikov_ic: pearson_correlation(&stoikov_preds, &realized),
    })
}

fn future_move_targets(
    observations: &[BenchmarkObservation],
    move_horizon: usize,
) -> Vec<Option<f64>> {
    let mut output = vec![None; observations.len()];
    if move_horizon == 0 || observations.is_empty() {
        return output;
    }

    let mut start = 0usize;
    while start < observations.len() {
        let sequence_id = observations[start].stoikov.sequence_id;
        let mut end = start + 1;
        while end < observations.len() && observations[end].stoikov.sequence_id == sequence_id {
            end += 1;
        }

        let price_change_indices = (start + 1..end)
            .filter(|&index| {
                observations[index].stoikov.mid_price != observations[index - 1].stoikov.mid_price
            })
            .collect::<Vec<_>>();

        if price_change_indices.len() >= move_horizon {
            for current_idx in start..end {
                let next_change_pos =
                    price_change_indices.partition_point(|&idx| idx <= current_idx);
                let target_pos = next_change_pos + move_horizon - 1;
                if target_pos < price_change_indices.len() {
                    let target_idx = price_change_indices[target_pos];
                    output[current_idx] = Some(
                        observations[target_idx].stoikov.mid_price
                            - observations[current_idx].stoikov.mid_price,
                    );
                }
            }
        }

        start = end;
    }

    output
}

fn future_time_targets(observations: &[BenchmarkObservation], horizon_us: i64) -> TimeTargetSet {
    let mut targets = vec![None; observations.len()];
    let mut move_counts = vec![None; observations.len()];
    if horizon_us <= 0 || observations.is_empty() {
        return TimeTargetSet {
            targets,
            move_counts,
        };
    }

    let mut start = 0usize;
    while start < observations.len() {
        let sequence_id = observations[start].stoikov.sequence_id;
        let mut end = start + 1;
        while end < observations.len() && observations[end].stoikov.sequence_id == sequence_id {
            end += 1;
        }

        let segment = &observations[start..end];
        let mut price_change_prefix = vec![0usize; segment.len() + 1];
        for index in 1..segment.len() {
            price_change_prefix[index + 1] = price_change_prefix[index]
                + usize::from(
                    segment[index].stoikov.mid_price != segment[index - 1].stoikov.mid_price,
                );
        }

        let mut target_offset = 1usize;
        for current_offset in 0..segment.len() {
            target_offset = target_offset.max(current_offset + 1);
            let target_time = segment[current_offset]
                .stoikov
                .event_time_us
                .saturating_add(horizon_us);

            while target_offset < segment.len()
                && segment[target_offset].stoikov.event_time_us < target_time
            {
                target_offset += 1;
            }

            if target_offset < segment.len() {
                targets[start + current_offset] = Some(
                    segment[target_offset].stoikov.mid_price
                        - segment[current_offset].stoikov.mid_price,
                );
                move_counts[start + current_offset] = Some(
                    price_change_prefix[target_offset + 1]
                        - price_change_prefix[current_offset + 1],
                );
            }
        }

        start = end;
    }

    TimeTargetSet {
        targets,
        move_counts,
    }
}

fn estimate_equivalent_move_horizon(
    observations: &[BenchmarkObservation],
    target_set: &TimeTargetSet,
    max_spread_ticks: u32,
) -> usize {
    let (sum_moves, count) = observations
        .iter()
        .zip(target_set.move_counts.iter())
        .filter(|(observation, move_count)| {
            observation.stoikov.spread_ticks <= max_spread_ticks && move_count.is_some()
        })
        .fold((0usize, 0usize), |(sum, count), (_, move_count)| {
            (sum + move_count.unwrap_or(0), count + 1)
        });

    if count == 0 {
        1
    } else {
        ((sum_moves as f64 / count as f64).round() as usize).max(1)
    }
}

fn format_time_label(horizon_ms: i64) -> String {
    if horizon_ms % 1_000 == 0 {
        format!("{}s", horizon_ms / 1_000)
    } else {
        format!("{horizon_ms}ms")
    }
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || y.len() != n {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;

    let (cov, var_x, var_y) = x.iter().zip(y.iter()).fold(
        (0.0_f64, 0.0_f64, 0.0_f64),
        |(cov, var_x, var_y), (&lhs, &rhs)| {
            let dx = lhs - mean_x;
            let dy = rhs - mean_y;
            (cov + dx * dy, var_x + dx * dx, var_y + dy * dy)
        },
    );

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 { 0.0 } else { cov / denom }
}
