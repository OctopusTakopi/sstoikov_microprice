use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StoikovObservation {
    pub sequence_id: u64,
    pub event_time_us: i64,
    pub mid_price: f64,
    pub imbalance: f64,
    pub spread_ticks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoikovMarkovConfig {
    pub num_imbalance_buckets: usize,
    pub min_spread_ticks: u32,
    pub max_spread_ticks: u32,
    pub symmetrize: bool,
    pub default_horizon_moves: usize,
    pub max_series_terms: usize,
    pub convergence_tol: f64,
}

impl Default for StoikovMarkovConfig {
    fn default() -> Self {
        Self {
            num_imbalance_buckets: 10,
            min_spread_ticks: 1,
            max_spread_ticks: 6,
            symmetrize: true,
            default_horizon_moves: 6,
            max_series_terms: 256,
            convergence_tol: 1e-12,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoikovTransitionRow {
    pub spread_ticks: u32,
    pub imbalance_bucket: usize,
    pub first_move_adjustment: f64,
    pub default_adjustment: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoikovMarkovMicroPrice {
    pub imbalance_bounds: Vec<f64>,
    pub min_spread_ticks: u32,
    pub max_spread_ticks: u32,
    pub price_step: f64,
    pub default_horizon_moves: usize,
    pub symmetrized: bool,
    pub first_move_adjustments: Vec<f64>,
    pub transition_after_move: Vec<Vec<f64>>,
    pub default_adjustments: Vec<f64>,
    pub series_terms_used: usize,
}

impl Default for StoikovMarkovMicroPrice {
    fn default() -> Self {
        Self::new()
    }
}

impl StoikovMarkovMicroPrice {
    pub fn new() -> Self {
        Self {
            imbalance_bounds: Vec::new(),
            min_spread_ticks: 1,
            max_spread_ticks: 1,
            price_step: 0.0,
            default_horizon_moves: 0,
            symmetrized: true,
            first_move_adjustments: Vec::new(),
            transition_after_move: Vec::new(),
            default_adjustments: Vec::new(),
            series_terms_used: 0,
        }
    }

    pub fn fit(
        &mut self,
        observations: &[StoikovObservation],
        tick_size: f64,
        config: &StoikovMarkovConfig,
    ) -> Result<(), StoikovMarkovError> {
        validate_config(tick_size, config)?;
        validate_observations(observations)?;

        let filtered_for_buckets = observations
            .iter()
            .filter(|obs| spread_in_range(obs.spread_ticks, config))
            .map(|obs| obs.imbalance)
            .collect::<Vec<_>>();
        if filtered_for_buckets.is_empty() {
            return Err(StoikovMarkovError::NoUsableObservations);
        }

        let mut sorted_imbalances = filtered_for_buckets;
        sorted_imbalances.sort_unstable_by(f64::total_cmp);
        let layout = build_quantile_layout(&sorted_imbalances, config.num_imbalance_buckets);
        let bucket_count = layout.bucket_count();
        let state_count =
            (config.max_spread_ticks - config.min_spread_ticks + 1) as usize * bucket_count;
        let price_step = tick_size / 2.0;

        let transitions =
            build_transition_counts(observations, &layout.bounds, price_step, config)?;

        let mut q = vec![vec![0.0; state_count]; state_count];
        let mut r2 = vec![vec![0.0; state_count]; state_count];
        let mut r1_by_step = vec![vec![0.0; transitions.move_steps.len()]; state_count];

        for (state_idx, total) in transitions.totals.iter().copied().enumerate() {
            if total == 0.0 {
                continue;
            }

            let scale = 1.0 / total;
            for next_state_idx in 0..state_count {
                q[state_idx][next_state_idx] =
                    transitions.q_counts[state_idx][next_state_idx] * scale;
                r2[state_idx][next_state_idx] =
                    transitions.r2_counts[state_idx][next_state_idx] * scale;
            }
            for move_idx in 0..transitions.move_steps.len() {
                r1_by_step[state_idx][move_idx] =
                    transitions.r1_counts[state_idx][move_idx] * scale;
            }
        }

        let move_values = transitions
            .move_steps
            .iter()
            .map(|step| *step as f64 * price_step)
            .collect::<Vec<_>>();

        let immediate_price_move = r1_by_step
            .iter()
            .map(|row| {
                row.iter()
                    .zip(move_values.iter())
                    .map(|(probability, move_value)| probability * move_value)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();

        let (first_move_adjustments, g1_terms) = accumulate_vector_series(
            &q,
            &immediate_price_move,
            config.max_series_terms,
            config.convergence_tol,
        );
        let (transition_after_move, b_terms) =
            accumulate_matrix_series(&q, &r2, config.max_series_terms, config.convergence_tol);
        let default_adjustments = adjustments_for_horizon_internal(
            &first_move_adjustments,
            &transition_after_move,
            config.default_horizon_moves,
        );

        self.imbalance_bounds = layout.bounds;
        self.min_spread_ticks = config.min_spread_ticks;
        self.max_spread_ticks = config.max_spread_ticks;
        self.price_step = price_step;
        self.default_horizon_moves = config.default_horizon_moves;
        self.symmetrized = config.symmetrize;
        self.first_move_adjustments = first_move_adjustments;
        self.transition_after_move = transition_after_move;
        self.default_adjustments = default_adjustments;
        self.series_terms_used = g1_terms.max(b_terms);
        self.validate()?;
        Ok(())
    }

    pub fn get_adjustment(&self, imbalance: f64, spread_ticks: u32) -> f64 {
        self.lookup_adjustment(&self.default_adjustments, imbalance, spread_ticks)
    }

    pub fn get_first_move_adjustment(&self, imbalance: f64, spread_ticks: u32) -> f64 {
        self.lookup_adjustment(&self.first_move_adjustments, imbalance, spread_ticks)
    }

    pub fn adjustments_for_horizon(&self, horizon_moves: usize) -> Vec<f64> {
        adjustments_for_horizon_internal(
            &self.first_move_adjustments,
            &self.transition_after_move,
            horizon_moves,
        )
    }

    pub fn get_adjustment_for_horizon(
        &self,
        imbalance: f64,
        spread_ticks: u32,
        horizon_moves: usize,
    ) -> f64 {
        let adjustments = self.adjustments_for_horizon(horizon_moves);
        self.lookup_adjustment(&adjustments, imbalance, spread_ticks)
    }

    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), StoikovMarkovError> {
        self.validate()?;
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Self, StoikovMarkovError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model: Self = serde_json::from_reader(reader)?;
        model.validate()?;
        Ok(model)
    }

    pub fn state_rows(&self) -> Vec<StoikovTransitionRow> {
        let bucket_count = self.bucket_count();
        let mut rows = Vec::with_capacity(self.state_count());
        for spread_ticks in self.min_spread_ticks..=self.max_spread_ticks {
            for bucket in 0..bucket_count {
                let idx = self.encode_state(spread_ticks, bucket);
                rows.push(StoikovTransitionRow {
                    spread_ticks,
                    imbalance_bucket: bucket,
                    first_move_adjustment: self
                        .first_move_adjustments
                        .get(idx)
                        .copied()
                        .unwrap_or(0.0),
                    default_adjustment: self.default_adjustments.get(idx).copied().unwrap_or(0.0),
                });
            }
        }
        rows
    }

    pub fn validate(&self) -> Result<(), StoikovMarkovError> {
        if self.price_step <= 0.0 || !self.price_step.is_finite() {
            return Err(StoikovMarkovError::InvalidPriceStep(self.price_step));
        }
        if self.max_spread_ticks < self.min_spread_ticks {
            return Err(StoikovMarkovError::InvalidSpreadRange {
                min_spread_ticks: self.min_spread_ticks,
                max_spread_ticks: self.max_spread_ticks,
            });
        }
        validate_bounds(&self.imbalance_bounds)?;

        let bucket_count = self.bucket_count();
        if bucket_count == 0 {
            return Err(StoikovMarkovError::InvalidBucketCount(0));
        }
        let state_count = self.state_count();
        if self.first_move_adjustments.len() != state_count {
            return Err(StoikovMarkovError::InvalidStateVectorLength {
                expected: state_count,
                actual: self.first_move_adjustments.len(),
            });
        }
        if self.default_adjustments.len() != state_count {
            return Err(StoikovMarkovError::InvalidStateVectorLength {
                expected: state_count,
                actual: self.default_adjustments.len(),
            });
        }
        if self.transition_after_move.len() != state_count {
            return Err(StoikovMarkovError::InvalidTransitionShape {
                expected_rows: state_count,
                actual_rows: self.transition_after_move.len(),
            });
        }
        for row in &self.transition_after_move {
            if row.len() != state_count {
                return Err(StoikovMarkovError::InvalidTransitionShape {
                    expected_rows: state_count,
                    actual_rows: row.len(),
                });
            }
        }

        for (index, value) in self
            .first_move_adjustments
            .iter()
            .chain(self.default_adjustments.iter())
            .enumerate()
        {
            if !value.is_finite() {
                return Err(StoikovMarkovError::NonFiniteAdjustment {
                    index,
                    value: *value,
                });
            }
        }
        for (row_idx, row) in self.transition_after_move.iter().enumerate() {
            for (col_idx, value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(StoikovMarkovError::NonFiniteTransition {
                        row: row_idx,
                        column: col_idx,
                        value: *value,
                    });
                }
            }
        }
        Ok(())
    }

    pub fn bucket_count(&self) -> usize {
        self.imbalance_bounds.len() + 1
    }

    pub fn state_count(&self) -> usize {
        (self.max_spread_ticks - self.min_spread_ticks + 1) as usize * self.bucket_count()
    }

    pub fn state_index(&self, imbalance: f64, spread_ticks: u32) -> Option<usize> {
        if !imbalance.is_finite() || self.first_move_adjustments.is_empty() {
            return None;
        }
        let clamped_spread = spread_ticks.clamp(self.min_spread_ticks, self.max_spread_ticks);
        let bucket = self
            .imbalance_bounds
            .partition_point(|&bound| bound < imbalance);
        Some(self.encode_state(clamped_spread, bucket))
    }

    fn lookup_adjustment(&self, adjustments: &[f64], imbalance: f64, spread_ticks: u32) -> f64 {
        self.state_index(imbalance, spread_ticks)
            .and_then(|idx| adjustments.get(idx).copied())
            .unwrap_or(0.0)
    }

    fn encode_state(&self, spread_ticks: u32, bucket: usize) -> usize {
        ((spread_ticks - self.min_spread_ticks) as usize) * self.bucket_count() + bucket
    }
}

#[derive(Debug, Error)]
pub enum StoikovMarkovError {
    #[error("tick size must be positive and finite, got {0}")]
    InvalidTickSize(f64),
    #[error("invalid bucket count {0}")]
    InvalidBucketCount(usize),
    #[error(
        "spread range is invalid: min_spread_ticks={min_spread_ticks}, max_spread_ticks={max_spread_ticks}"
    )]
    InvalidSpreadRange {
        min_spread_ticks: u32,
        max_spread_ticks: u32,
    },
    #[error("price step must be positive and finite, got {0}")]
    InvalidPriceStep(f64),
    #[error("need at least two observations to estimate transitions")]
    TooFewObservations,
    #[error("no observations fall inside the configured spread range")]
    NoUsableObservations,
    #[error("no valid transitions were observed for the configured state space")]
    NoTransitions,
    #[error("observation {index} has non-finite {field}: {value}")]
    NonFiniteObservation {
        index: usize,
        field: &'static str,
        value: f64,
    },
    #[error("imbalance bound at index {index} is not finite: {value}")]
    NonFiniteBound { index: usize, value: f64 },
    #[error(
        "imbalance bounds must be strictly increasing, but bounds[{previous_index}]={previous_value} and bounds[{index}]={value}"
    )]
    BoundsNotStrictlyIncreasing {
        previous_index: usize,
        previous_value: f64,
        index: usize,
        value: f64,
    },
    #[error("invalid state vector length: expected {expected}, got {actual}")]
    InvalidStateVectorLength { expected: usize, actual: usize },
    #[error(
        "invalid transition shape: expected square matrix with {expected_rows} rows/cols, got {actual_rows}"
    )]
    InvalidTransitionShape {
        expected_rows: usize,
        actual_rows: usize,
    },
    #[error("non-finite adjustment at index {index}: {value}")]
    NonFiniteAdjustment { index: usize, value: f64 },
    #[error("non-finite transition at row {row}, column {column}: {value}")]
    NonFiniteTransition {
        row: usize,
        column: usize,
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

struct TransitionCounts {
    q_counts: Vec<Vec<f64>>,
    r1_counts: Vec<Vec<f64>>,
    r2_counts: Vec<Vec<f64>>,
    totals: Vec<f64>,
    move_steps: Vec<i32>,
}

fn validate_config(tick_size: f64, config: &StoikovMarkovConfig) -> Result<(), StoikovMarkovError> {
    if !tick_size.is_finite() || tick_size <= 0.0 {
        return Err(StoikovMarkovError::InvalidTickSize(tick_size));
    }
    if config.num_imbalance_buckets == 0 {
        return Err(StoikovMarkovError::InvalidBucketCount(0));
    }
    if config.max_spread_ticks < config.min_spread_ticks {
        return Err(StoikovMarkovError::InvalidSpreadRange {
            min_spread_ticks: config.min_spread_ticks,
            max_spread_ticks: config.max_spread_ticks,
        });
    }
    Ok(())
}

fn validate_observations(observations: &[StoikovObservation]) -> Result<(), StoikovMarkovError> {
    if observations.len() < 2 {
        return Err(StoikovMarkovError::TooFewObservations);
    }
    for (index, observation) in observations.iter().enumerate() {
        if !observation.mid_price.is_finite() {
            return Err(StoikovMarkovError::NonFiniteObservation {
                index,
                field: "mid_price",
                value: observation.mid_price,
            });
        }
        if !observation.imbalance.is_finite() {
            return Err(StoikovMarkovError::NonFiniteObservation {
                index,
                field: "imbalance",
                value: observation.imbalance,
            });
        }
    }
    Ok(())
}

fn validate_bounds(bounds: &[f64]) -> Result<(), StoikovMarkovError> {
    for (index, &bound) in bounds.iter().enumerate() {
        if !bound.is_finite() {
            return Err(StoikovMarkovError::NonFiniteBound {
                index,
                value: bound,
            });
        }
        if index > 0 && bounds[index - 1] >= bound {
            return Err(StoikovMarkovError::BoundsNotStrictlyIncreasing {
                previous_index: index - 1,
                previous_value: bounds[index - 1],
                index,
                value: bound,
            });
        }
    }
    Ok(())
}

fn spread_in_range(spread_ticks: u32, config: &StoikovMarkovConfig) -> bool {
    spread_ticks >= config.min_spread_ticks && spread_ticks <= config.max_spread_ticks
}

fn build_transition_counts(
    observations: &[StoikovObservation],
    bounds: &[f64],
    price_step: f64,
    config: &StoikovMarkovConfig,
) -> Result<TransitionCounts, StoikovMarkovError> {
    let bucket_count = bounds.len() + 1;
    let state_count =
        (config.max_spread_ticks - config.min_spread_ticks + 1) as usize * bucket_count;

    let mut raw_transitions = Vec::new();
    let mut move_steps = Vec::new();

    for pair in observations.windows(2) {
        let current = pair[0];
        let next = pair[1];
        if current.sequence_id != next.sequence_id {
            continue;
        }
        if !spread_in_range(current.spread_ticks, config)
            || !spread_in_range(next.spread_ticks, config)
        {
            continue;
        }

        let current_bucket = bounds.partition_point(|&bound| bound < current.imbalance);
        let next_bucket = bounds.partition_point(|&bound| bound < next.imbalance);
        let current_state = encode_state(
            current.spread_ticks,
            current_bucket,
            config.min_spread_ticks,
            bucket_count,
        );
        let next_state = encode_state(
            next.spread_ticks,
            next_bucket,
            config.min_spread_ticks,
            bucket_count,
        );
        let delta_steps = ((next.mid_price - current.mid_price) / price_step).round() as i32;
        raw_transitions.push((
            current_state,
            next_state,
            current_bucket,
            next_bucket,
            delta_steps,
        ));
        if delta_steps != 0 && !move_steps.contains(&delta_steps) {
            move_steps.push(delta_steps);
        }
        if config.symmetrize && delta_steps != 0 && !move_steps.contains(&-delta_steps) {
            move_steps.push(-delta_steps);
        }
    }

    if raw_transitions.is_empty() {
        return Err(StoikovMarkovError::NoTransitions);
    }

    move_steps.sort_unstable();
    let mut q_counts = vec![vec![0.0; state_count]; state_count];
    let mut r2_counts = vec![vec![0.0; state_count]; state_count];
    let mut r1_counts = vec![vec![0.0; move_steps.len()]; state_count];
    let mut totals = vec![0.0; state_count];

    for &(current_state, next_state, current_bucket, next_bucket, delta_steps) in &raw_transitions {
        add_transition_counts(
            current_state,
            next_state,
            delta_steps,
            &move_steps,
            &mut q_counts,
            &mut r1_counts,
            &mut r2_counts,
            &mut totals,
        );

        if config.symmetrize {
            let current_spread = config.min_spread_ticks + (current_state / bucket_count) as u32;
            let next_spread = config.min_spread_ticks + (next_state / bucket_count) as u32;
            let mirrored_current = encode_state(
                current_spread,
                bucket_count - 1 - current_bucket,
                config.min_spread_ticks,
                bucket_count,
            );
            let mirrored_next = encode_state(
                next_spread,
                bucket_count - 1 - next_bucket,
                config.min_spread_ticks,
                bucket_count,
            );
            add_transition_counts(
                mirrored_current,
                mirrored_next,
                -delta_steps,
                &move_steps,
                &mut q_counts,
                &mut r1_counts,
                &mut r2_counts,
                &mut totals,
            );
        }
    }

    if totals.iter().all(|&total| total == 0.0) {
        return Err(StoikovMarkovError::NoTransitions);
    }

    Ok(TransitionCounts {
        q_counts,
        r1_counts,
        r2_counts,
        totals,
        move_steps,
    })
}

fn add_transition_counts(
    current_state: usize,
    next_state: usize,
    delta_steps: i32,
    move_steps: &[i32],
    q_counts: &mut [Vec<f64>],
    r1_counts: &mut [Vec<f64>],
    r2_counts: &mut [Vec<f64>],
    totals: &mut [f64],
) {
    totals[current_state] += 1.0;
    if delta_steps == 0 {
        q_counts[current_state][next_state] += 1.0;
    } else if let Ok(move_idx) = move_steps.binary_search(&delta_steps) {
        r1_counts[current_state][move_idx] += 1.0;
        r2_counts[current_state][next_state] += 1.0;
    }
}

fn encode_state(
    spread_ticks: u32,
    bucket: usize,
    min_spread_ticks: u32,
    bucket_count: usize,
) -> usize {
    ((spread_ticks - min_spread_ticks) as usize) * bucket_count + bucket
}

fn accumulate_vector_series(
    q: &[Vec<f64>],
    seed: &[f64],
    max_terms: usize,
    tolerance: f64,
) -> (Vec<f64>, usize) {
    if seed.is_empty() {
        return (Vec::new(), 0);
    }

    let mut sum = seed.to_vec();
    let mut term = seed.to_vec();
    let mut terms_used = 1usize;

    for _ in 1..max_terms {
        term = mat_vec_mul(q, &term);
        if max_abs_vec(&term) <= tolerance {
            break;
        }
        add_vec_in_place(&mut sum, &term);
        terms_used += 1;
    }

    (sum, terms_used)
}

fn accumulate_matrix_series(
    q: &[Vec<f64>],
    seed: &[Vec<f64>],
    max_terms: usize,
    tolerance: f64,
) -> (Vec<Vec<f64>>, usize) {
    if seed.is_empty() {
        return (Vec::new(), 0);
    }

    let mut sum = seed.to_vec();
    let mut term = seed.to_vec();
    let mut terms_used = 1usize;

    for _ in 1..max_terms {
        term = mat_mul(q, &term);
        if max_abs_mat(&term) <= tolerance {
            break;
        }
        add_mat_in_place(&mut sum, &term);
        terms_used += 1;
    }

    (sum, terms_used)
}

fn adjustments_for_horizon_internal(
    first_move_adjustments: &[f64],
    transition_after_move: &[Vec<f64>],
    horizon_moves: usize,
) -> Vec<f64> {
    if horizon_moves == 0 || first_move_adjustments.is_empty() {
        return vec![0.0; first_move_adjustments.len()];
    }

    let mut adjustments = first_move_adjustments.to_vec();
    let mut tail = first_move_adjustments.to_vec();

    for _ in 1..horizon_moves {
        tail = mat_vec_mul(transition_after_move, &tail);
        add_vec_in_place(&mut adjustments, &tail);
    }

    adjustments
}

fn mat_vec_mul(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector.iter())
                .map(|(left, right)| left * right)
                .sum::<f64>()
        })
        .collect()
}

fn mat_mul(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if left.is_empty() || right.is_empty() {
        return Vec::new();
    }

    let rows = left.len();
    let cols = right[0].len();
    let mut output = vec![vec![0.0; cols]; rows];

    for (row_idx, left_row) in left.iter().enumerate() {
        for (mid_idx, &left_value) in left_row.iter().enumerate() {
            if left_value == 0.0 {
                continue;
            }
            for col_idx in 0..cols {
                output[row_idx][col_idx] += left_value * right[mid_idx][col_idx];
            }
        }
    }

    output
}

fn add_vec_in_place(target: &mut [f64], source: &[f64]) {
    for (target_value, source_value) in target.iter_mut().zip(source.iter()) {
        *target_value += source_value;
    }
}

fn add_mat_in_place(target: &mut [Vec<f64>], source: &[Vec<f64>]) {
    for (target_row, source_row) in target.iter_mut().zip(source.iter()) {
        add_vec_in_place(target_row, source_row);
    }
}

fn max_abs_vec(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn max_abs_mat(values: &[Vec<f64>]) -> f64 {
    values
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.abs())
        .fold(0.0, f64::max)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_model_path(prefix: &str) -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("valid clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "stoikov_markov_{prefix}_{}_{}.json",
            std::process::id(),
            timestamp
        ))
    }

    #[test]
    fn fits_simple_two_state_chain() {
        let observations = vec![
            StoikovObservation {
                sequence_id: 0,
                event_time_us: 0,
                mid_price: 100.0,
                imbalance: 0.1,
                spread_ticks: 1,
            },
            StoikovObservation {
                sequence_id: 0,
                event_time_us: 1,
                mid_price: 100.0,
                imbalance: 0.1,
                spread_ticks: 1,
            },
            StoikovObservation {
                sequence_id: 0,
                event_time_us: 2,
                mid_price: 99.5,
                imbalance: 0.1,
                spread_ticks: 1,
            },
            StoikovObservation {
                sequence_id: 1,
                event_time_us: 3,
                mid_price: 100.0,
                imbalance: 0.9,
                spread_ticks: 1,
            },
            StoikovObservation {
                sequence_id: 1,
                event_time_us: 4,
                mid_price: 100.0,
                imbalance: 0.9,
                spread_ticks: 1,
            },
            StoikovObservation {
                sequence_id: 1,
                event_time_us: 5,
                mid_price: 100.5,
                imbalance: 0.9,
                spread_ticks: 1,
            },
        ];

        let mut model = StoikovMarkovMicroPrice::new();
        let config = StoikovMarkovConfig {
            num_imbalance_buckets: 2,
            min_spread_ticks: 1,
            max_spread_ticks: 1,
            symmetrize: false,
            default_horizon_moves: 3,
            max_series_terms: 128,
            convergence_tol: 1e-15,
        };

        model.fit(&observations, 1.0, &config).unwrap();

        let first = model.first_move_adjustments.clone();
        assert_eq!(first.len(), 2);
        assert!((first[0] + 0.5).abs() < 1e-12, "{first:?}");
        assert!((first[1] - 0.5).abs() < 1e-12, "{first:?}");

        let horizon_three = model.adjustments_for_horizon(3);
        assert!((horizon_three[0] + 1.5).abs() < 1e-12, "{horizon_three:?}");
        assert!((horizon_three[1] - 1.5).abs() < 1e-12, "{horizon_three:?}");
        assert!((model.get_adjustment(0.1, 1) + 1.5).abs() < 1e-12);
        assert!((model.get_adjustment(0.9, 1) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn load_and_save_round_trip() {
        let path = temp_model_path("round_trip");
        let observations = vec![
            StoikovObservation {
                sequence_id: 0,
                event_time_us: 0,
                mid_price: 100.0,
                imbalance: 0.2,
                spread_ticks: 1,
            },
            StoikovObservation {
                sequence_id: 0,
                event_time_us: 1,
                mid_price: 100.5,
                imbalance: 0.8,
                spread_ticks: 1,
            },
            StoikovObservation {
                sequence_id: 0,
                event_time_us: 2,
                mid_price: 100.5,
                imbalance: 0.8,
                spread_ticks: 1,
            },
        ];
        let mut model = StoikovMarkovMicroPrice::new();
        model
            .fit(&observations, 1.0, &StoikovMarkovConfig::default())
            .unwrap();
        model.save_model(&path).unwrap();

        let loaded = StoikovMarkovMicroPrice::load_model(&path).unwrap();
        assert_eq!(loaded.imbalance_bounds, model.imbalance_bounds);
        assert_eq!(loaded.first_move_adjustments, model.first_move_adjustments);
        assert_eq!(loaded.default_adjustments, model.default_adjustments);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_non_finite_observation() {
        let observations = vec![
            StoikovObservation {
                sequence_id: 0,
                event_time_us: 0,
                mid_price: 100.0,
                imbalance: 0.2,
                spread_ticks: 1,
            },
            StoikovObservation {
                sequence_id: 0,
                event_time_us: 1,
                mid_price: f64::NAN,
                imbalance: 0.8,
                spread_ticks: 1,
            },
        ];

        let mut model = StoikovMarkovMicroPrice::new();
        let error = model
            .fit(&observations, 1.0, &StoikovMarkovConfig::default())
            .unwrap_err();
        assert!(matches!(
            error,
            StoikovMarkovError::NonFiniteObservation {
                field: "mid_price",
                ..
            }
        ));
    }
}
