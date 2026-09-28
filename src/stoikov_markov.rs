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
}

impl Default for StoikovMarkovConfig {
    fn default() -> Self {
        Self {
            num_imbalance_buckets: 10,
            min_spread_ticks: 1,
            max_spread_ticks: 6,
            symmetrize: true,
            default_horizon_moves: 6,
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
}

/// Limit of [`StoikovMarkovMicroPrice::adjustments_for_horizon`] as the horizon grows:
/// G* = sum_k B^k G1. `adjustments` always equals `adjustments_for_horizon(moves)`.
#[derive(Debug, Clone, PartialEq)]
pub enum MoveSeriesLimit {
    /// The expected size of move `moves + 1`, given the chain is still on states observed
    /// as current, was within tolerance, so it and the rest of the series were dropped.
    Converged { adjustments: Vec<f64>, moves: usize },
    /// `moves` terms were summed without reaching tolerance, e.g. an unsymmetrized chain
    /// with drift.
    Capped { adjustments: Vec<f64>, moves: usize },
    /// The terms shrank only because the chain drained into states never observed as
    /// current (zero rows of B): after `moves` moves at most [`EXHAUSTED_MASS`] of the
    /// probability is left, so the partial sum is not a limit.
    Leaked { adjustments: Vec<f64>, moves: usize },
}

impl MoveSeriesLimit {
    pub fn adjustments(&self) -> &[f64] {
        match self {
            Self::Converged { adjustments, .. }
            | Self::Capped { adjustments, .. }
            | Self::Leaked { adjustments, .. } => adjustments,
        }
    }

    pub fn moves(&self) -> usize {
        match self {
            Self::Converged { moves, .. }
            | Self::Capped { moves, .. }
            | Self::Leaked { moves, .. } => *moves,
        }
    }

    pub fn is_converged(&self) -> bool {
        matches!(self, Self::Converged { .. })
    }
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
        }
    }

    /// Fits bounds, G1, B and G_{default_horizon_moves}. With `config.symmetrize` the bucket
    /// layout is mirror-symmetric, which can cost one bucket: when the tie group at exactly 0.5
    /// straddles the centre, one fewer bucket keeps it whole. Fails with
    /// [`StoikovMarkovError::NonMovingClosedClass`] if the kept transitions contain a closed
    /// set of states that never moves.
    pub fn fit(
        &mut self,
        observations: &[StoikovObservation],
        tick_size: f64,
        config: &StoikovMarkovConfig,
    ) -> Result<(), StoikovMarkovError> {
        validate_config(tick_size, config)?;
        validate_observations(observations)?;

        // Stoikov (2018) symmetrizes the data before binning: quantiles come from the pooled
        // sample {I} U {1 - I}, so mirrored imbalances bin with the same bounds.
        let mut sorted_imbalances = observations
            .iter()
            .filter(|obs| spread_in_range(obs.spread_ticks, config))
            .map(|obs| obs.imbalance)
            .collect::<Vec<_>>();
        if sorted_imbalances.is_empty() {
            return Err(StoikovMarkovError::NoUsableObservations);
        }
        sorted_imbalances.sort_unstable_by(f64::total_cmp);
        let imbalance_bounds = if config.symmetrize {
            let pooled = merge_with_mirror(&sorted_imbalances);
            drop(sorted_imbalances);
            build_symmetric_quantile_bounds(&pooled, config.num_imbalance_buckets)
        } else {
            build_quantile_bounds(&sorted_imbalances, config.num_imbalance_buckets)
        };
        let price_step = tick_size / 2.0;

        let chain = FirstMoveChain::estimate(observations, &imbalance_bounds, price_step, config)?;
        let (first_move_adjustments, transition_after_move) = chain.solve()?;
        let default_adjustments = horizon_adjustments(
            &first_move_adjustments,
            &transition_after_move,
            config.default_horizon_moves,
        );

        self.imbalance_bounds = imbalance_bounds;
        self.min_spread_ticks = config.min_spread_ticks;
        self.max_spread_ticks = config.max_spread_ticks;
        self.price_step = price_step;
        self.default_horizon_moves = config.default_horizon_moves;
        self.symmetrized = config.symmetrize;
        self.first_move_adjustments = first_move_adjustments;
        self.transition_after_move = transition_after_move;
        self.default_adjustments = default_adjustments;
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
        horizon_adjustments(
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

    /// The limit microprice adjustment G* = sum_k B^k G1. A state never observed as current
    /// has a zero B row, so B can be substochastic and B^k G1 shrink with the surviving mass
    /// m_k = B^k 1 rather than with the drift. Term k therefore stops the sum as
    /// `Converged` when |B^k G1| <= `tolerance` * m_k in every state (some mass left), as
    /// `Leaked` when every m_k <= [`EXHAUSTED_MASS`], and as `Capped` at `max_moves` terms.
    pub fn limit_adjustments(&self, tolerance: f64, max_moves: usize) -> MoveSeriesLimit {
        if self.first_move_adjustments.is_empty() {
            return MoveSeriesLimit::Converged {
                adjustments: Vec::new(),
                moves: 0,
            };
        }
        let mut outcome = None;
        let (adjustments, moves) = sum_move_series(
            &self.first_move_adjustments,
            &self.transition_after_move,
            // Column 0 carries B^k G1, column 1 the surviving mass B^k 1.
            |g1| [g1, 1.0],
            |moves, terms| {
                let converged = terms.iter().any(|&[_, mass]| mass > 0.0)
                    && terms
                        .iter()
                        .all(|&[value, mass]| value.abs() <= tolerance * mass);
                outcome = if converged {
                    Some(SeriesStop::Converged)
                } else if terms.iter().all(|&[_, mass]| mass <= EXHAUSTED_MASS) {
                    Some(SeriesStop::Leaked)
                } else if moves == max_moves {
                    Some(SeriesStop::Capped)
                } else {
                    None
                };
                outcome.is_some()
            },
        );
        match outcome.expect("sum_move_series stops only when an outcome is set") {
            SeriesStop::Converged => MoveSeriesLimit::Converged { adjustments, moves },
            SeriesStop::Capped => MoveSeriesLimit::Capped { adjustments, moves },
            SeriesStop::Leaked => MoveSeriesLimit::Leaked { adjustments, moves },
        }
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
    /// Every observed transition out of some set of states stays in the set without moving
    /// the mid, so the first move from it is undefined. Besides a genuinely frozen book, the
    /// spread filter can cause this: a rare in-range state whose only kept transitions are
    /// no-move self-loops (its other transitions left the spread range).
    #[error(
        "I - Q is singular at state {state}: a closed set of states never moves the mid price \
         (widen the spread range or use fewer buckets)"
    )]
    NonMovingClosedClass { state: usize },
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

/// Pooled values closer than this are one tie group: 1 - (1 - I) can differ from I by an
/// ulp of 1, and a mirrored ratio must share its bucket with the raw one (e.g. 1 - 0.8 vs 0.2).
const MIRROR_TIE_TOLERANCE: f64 = 4.0 * f64::EPSILON;

/// Pivots of I - Q at or below this are treated as zero. Real pivots are of the order of
/// a state's probability of escaping without returning, far above this for any sample.
const MIN_PIVOT: f64 = 1e-12;

/// Surviving probability at or below which [`MoveSeriesLimit::Leaked`] is reported.
pub const EXHAUSTED_MASS: f64 = 1e-12;

enum SeriesStop {
    Converged,
    Capped,
    Leaked,
}

/// First-step law of the event chain, row-major over `state_count` states.
struct FirstMoveChain {
    state_count: usize,
    /// Q: probability of the next state without a mid change.
    no_move: Vec<f64>,
    /// t: expected mid change of the first step.
    expected_move: Vec<f64>,
    /// R2: probability of a mid change landing in the next state.
    after_move: Vec<f64>,
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

impl FirstMoveChain {
    /// Empirical Q, t and R2. With `config.symmetrize` every transition is also counted
    /// mirrored, (1 - I, S) -> (1 - I', S') with the opposite mid change, binned by value
    /// with the same bounds.
    fn estimate(
        observations: &[StoikovObservation],
        bounds: &[f64],
        price_step: f64,
        config: &StoikovMarkovConfig,
    ) -> Result<Self, StoikovMarkovError> {
        let bucket_count = bounds.len() + 1;
        let state_count =
            (config.max_spread_ticks - config.min_spread_ticks + 1) as usize * bucket_count;
        let state = |imbalance: f64, spread_ticks: u32| {
            let bucket = bounds.partition_point(|&bound| bound < imbalance);
            encode_state(spread_ticks, bucket, config.min_spread_ticks, bucket_count)
        };

        let mut chain = Self {
            state_count,
            no_move: vec![0.0; state_count * state_count],
            expected_move: vec![0.0; state_count],
            after_move: vec![0.0; state_count * state_count],
        };
        let mut totals = vec![0.0; state_count];
        let mut record = |current: usize, next: usize, delta_steps: i32| {
            totals[current] += 1.0;
            let cell = current * state_count + next;
            if delta_steps == 0 {
                chain.no_move[cell] += 1.0;
            } else {
                chain.expected_move[current] += delta_steps as f64 * price_step;
                chain.after_move[cell] += 1.0;
            }
        };

        // Consecutive in-range observations of one sequence. Each observation is binned once
        // (raw and mirrored) and its states are carried to the pair it starts.
        let mut previous: Option<(&StoikovObservation, usize, Option<usize>)> = None;
        for next in observations {
            let binned = spread_in_range(next.spread_ticks, config).then(|| {
                let raw = state(next.imbalance, next.spread_ticks);
                let mirrored = config
                    .symmetrize
                    .then(|| state(1.0 - next.imbalance, next.spread_ticks));
                (next, raw, mirrored)
            });
            if let (Some((current, from, from_mirrored)), Some((_, to, to_mirrored))) =
                (previous, binned)
                && current.sequence_id == next.sequence_id
            {
                let delta_steps =
                    ((next.mid_price - current.mid_price) / price_step).round() as i32;
                record(from, to, delta_steps);
                if let (Some(from), Some(to)) = (from_mirrored, to_mirrored) {
                    record(from, to, -delta_steps);
                }
            }
            previous = binned;
        }

        if totals.iter().all(|&total| total == 0.0) {
            return Err(StoikovMarkovError::NoTransitions);
        }

        for (state_idx, &total) in totals.iter().enumerate() {
            if total == 0.0 {
                continue;
            }
            let scale = 1.0 / total;
            let row = state_idx * state_count..(state_idx + 1) * state_count;
            chain.no_move[row.clone()]
                .iter_mut()
                .chain(&mut chain.after_move[row])
                .for_each(|value| *value *= scale);
            chain.expected_move[state_idx] *= scale;
        }
        Ok(chain)
    }

    /// G1 = (I - Q)^-1 t and B = (I - Q)^-1 R2, from one elimination on [t | R2]. A state
    /// never observed as current has a zero Q row, i.e. an identity row in I - Q.
    fn solve(&self) -> Result<(Vec<f64>, Vec<Vec<f64>>), StoikovMarkovError> {
        let n = self.state_count;
        let width = n + 1;
        let mut lhs = self.no_move.iter().map(|q| -q).collect::<Vec<_>>();
        for diagonal in lhs.iter_mut().step_by(n + 1) {
            *diagonal += 1.0;
        }
        let mut rhs = vec![0.0; n * width];
        for ((row, &t), r2) in rhs
            .chunks_exact_mut(width)
            .zip(&self.expected_move)
            .zip(self.after_move.chunks_exact(n))
        {
            row[0] = t;
            row[1..].copy_from_slice(r2);
        }

        solve_in_place(&mut lhs, &mut rhs, n)?;

        let first_move_adjustments = rhs.chunks_exact(width).map(|row| row[0]).collect();
        let transition_after_move = rhs
            .chunks_exact(width)
            .map(|row| row[1..].to_vec())
            .collect();
        Ok((first_move_adjustments, transition_after_move))
    }
}

/// Solves `lhs * X = rhs` in place (X lands in `rhs`) by Gaussian elimination with partial
/// pivoting; `lhs` is n x n and `rhs` n x w, both row-major.
fn solve_in_place(lhs: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), StoikovMarkovError> {
    debug_assert_eq!(lhs.len(), n * n);
    let width = rhs.len() / n;

    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&left, &right| {
                lhs[left * n + col]
                    .abs()
                    .total_cmp(&lhs[right * n + col].abs())
            })
            .expect("col < n");
        let pivot = lhs[pivot_row * n + col];
        if pivot.abs() <= MIN_PIVOT {
            return Err(StoikovMarkovError::NonMovingClosedClass { state: col });
        }
        if pivot_row != col {
            swap_rows(lhs, n, col, pivot_row);
            swap_rows(rhs, width, col, pivot_row);
        }
        for row in col + 1..n {
            let factor = lhs[row * n + col] / pivot;
            if factor != 0.0 {
                subtract_scaled_row(lhs, n, col, row, factor, col);
                subtract_scaled_row(rhs, width, col, row, factor, 0);
            }
        }
    }

    for row in (0..n).rev() {
        let (head, solved) = rhs.split_at_mut((row + 1) * width);
        let target = &mut head[row * width..];
        for (coefficient, solution) in lhs[row * n + row + 1..(row + 1) * n]
            .iter()
            .zip(solved.chunks_exact(width))
        {
            if *coefficient != 0.0 {
                for (value, x) in target.iter_mut().zip(solution) {
                    *value -= coefficient * x;
                }
            }
        }
        let inverse_pivot = 1.0 / lhs[row * n + row];
        target.iter_mut().for_each(|value| *value *= inverse_pivot);
    }
    Ok(())
}

/// Swaps rows `upper < lower` of a row-major matrix.
fn swap_rows(matrix: &mut [f64], width: usize, upper: usize, lower: usize) {
    let (head, tail) = matrix.split_at_mut(lower * width);
    head[upper * width..(upper + 1) * width].swap_with_slice(&mut tail[..width]);
}

/// `row -= factor * pivot_row` over columns `from_col..`, for `pivot_row < row`.
fn subtract_scaled_row(
    matrix: &mut [f64],
    width: usize,
    pivot_row: usize,
    row: usize,
    factor: f64,
    from_col: usize,
) {
    let (head, tail) = matrix.split_at_mut(row * width);
    let pivot = &head[pivot_row * width + from_col..(pivot_row + 1) * width];
    for (value, pivot_value) in tail[from_col..width].iter_mut().zip(pivot) {
        *value -= factor * pivot_value;
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

/// G_h = sum_{k < h} B^k G1.
fn horizon_adjustments(
    first_move_adjustments: &[f64],
    transition_after_move: &[Vec<f64>],
    horizon_moves: usize,
) -> Vec<f64> {
    sum_move_series(
        first_move_adjustments,
        transition_after_move,
        |g1| [g1],
        |moves, _| moves == horizon_moves,
    )
    .0
}

/// Sums the terms B^k G1, k = 0, 1, ..., stopping before the first k for which
/// `stop(k, B^k C)` holds, where C = `column(G1)` row by row: column 0 of C must be G1 and
/// the other W - 1 columns are propagated alongside at no extra pass. Returns the partial
/// sum and the number of terms summed.
fn sum_move_series<const W: usize>(
    first_move_adjustments: &[f64],
    transition_after_move: &[Vec<f64>],
    column: impl Fn(f64) -> [f64; W],
    mut stop: impl FnMut(usize, &[[f64; W]]) -> bool,
) -> (Vec<f64>, usize) {
    let n = first_move_adjustments.len();
    let mut sum = vec![0.0; n];
    let mut terms = first_move_adjustments
        .iter()
        .map(|&g1| column(g1))
        .collect::<Vec<_>>();
    let mut next_terms = vec![[0.0; W]; n];
    let mut moves = 0usize;
    while !stop(moves, &terms) {
        for (total, term) in sum.iter_mut().zip(&terms) {
            *total += term[0];
        }
        for (out, row) in next_terms.iter_mut().zip(transition_after_move) {
            *out = row.iter().zip(&terms).fold([0.0; W], |mut acc, (b, term)| {
                acc.iter_mut()
                    .zip(term)
                    .for_each(|(a, value)| *a += b * value);
                acc
            });
        }
        std::mem::swap(&mut terms, &mut next_terms);
        moves += 1;
    }
    (sum, moves)
}

/// The pooled sample {I} U {1 - I} sorted by `total_cmp`, from the sorted finite `sorted`.
/// Rounding is monotone, so the mirrors 1 - sorted[n - 1 - j] ascend with j and never
/// produce -0.0; a merge of the two runs is therefore the (unique) sorted pooled sample.
fn merge_with_mirror(sorted: &[f64]) -> Vec<f64> {
    let mut pooled = Vec::with_capacity(2 * sorted.len());
    let mut raw = sorted.iter().copied().peekable();
    let mut mirrored = sorted
        .iter()
        .rev()
        .map(|&imbalance| 1.0 - imbalance)
        .peekable();
    while let (Some(&value), Some(&mirror)) = (raw.peek(), mirrored.peek()) {
        pooled.push(if value.total_cmp(&mirror).is_le() {
            raw.next();
            value
        } else {
            mirrored.next();
            mirror
        });
    }
    pooled.extend(raw.chain(mirrored));
    pooled
}

/// End positions (exclusive) of the tie groups of a sorted sample: maximal runs whose
/// consecutive gaps are at most `tie_tolerance`. The last entry is `sorted.len()`.
fn tie_group_ends(sorted: &[f64], tie_tolerance: f64) -> Vec<usize> {
    let mut ends = (1..sorted.len())
        .filter(|&index| sorted[index] - sorted[index - 1] > tie_tolerance)
        .collect::<Vec<_>>();
    ends.push(sorted.len());
    ends
}

/// Picks `count` increasing cut positions from the sorted `candidates`, cut k (1-based)
/// nearest to k * total / denominator while leaving a candidate for every later cut.
fn pick_cuts(candidates: &[usize], count: usize, total: usize, denominator: usize) -> Vec<usize> {
    debug_assert!(count <= candidates.len());
    let scaled = |index: usize| candidates[index] as u128 * denominator as u128;
    let mut cuts = Vec::with_capacity(count);
    let mut first = 0usize;
    for k in 1..=count {
        let target = k as u128 * total as u128;
        let last = candidates.len() - 1 - (count - k);
        let mut index = first;
        while index < last && scaled(index) < target {
            index += 1;
        }
        if index > first && target.abs_diff(scaled(index - 1)) <= target.abs_diff(scaled(index)) {
            index -= 1;
        }
        cuts.push(candidates[index]);
        first = index + 1;
    }
    cuts
}

/// Tie-aware quantile bounds: equal values are never split across buckets, and each bound
/// is the largest value of its bucket.
fn build_quantile_bounds(sorted_imbalances: &[f64], requested_buckets: usize) -> Vec<f64> {
    debug_assert!(!sorted_imbalances.is_empty());
    debug_assert!(requested_buckets > 0);
    let group_ends = tie_group_ends(sorted_imbalances, 0.0);
    let candidates = &group_ends[..group_ends.len() - 1];
    let count = (requested_buckets - 1).min(candidates.len());
    pick_cuts(candidates, count, sorted_imbalances.len(), count + 1)
        .into_iter()
        .map(|cut| sorted_imbalances[cut - 1])
        .collect()
}

/// Tie-aware quantile bounds of the pooled sample {I} U {1 - I} (sorted, length 2N), cut at
/// mirror-image positions c and 2N - c so that I -> 1 - I maps bucket k exactly onto bucket
/// n - 1 - k. An even n needs a cut at the centre N; when a tie group straddles it (exact
/// 0.5 imbalances, which mirror onto themselves) n - 1 buckets are used instead and the
/// middle one holds that group.
fn build_symmetric_quantile_bounds(sorted_pooled: &[f64], requested_buckets: usize) -> Vec<f64> {
    debug_assert!(!sorted_pooled.is_empty() && sorted_pooled.len().is_multiple_of(2));
    debug_assert!(requested_buckets > 0);
    let total = sorted_pooled.len();
    let half = total / 2;
    // Tie-group end test of `tie_group_ends(sorted_pooled, MIRROR_TIE_TOLERANCE)`, per position.
    let is_group_end = |position: usize| {
        position == total
            || (position > 0
                && position < total
                && sorted_pooled[position] - sorted_pooled[position - 1] > MIRROR_TIE_TOLERANCE)
    };

    let centre_cut = requested_buckets.is_multiple_of(2) && is_group_end(half);
    let buckets = if !requested_buckets.is_multiple_of(2) || centre_cut {
        requested_buckets
    } else {
        requested_buckets - 1
    };
    // Lower-half cuts whose mirror position is a group end too (it always is up to the
    // rounding of 1 - I, which the tie tolerance absorbs).
    let candidates = (1..half)
        .filter(|&end| is_group_end(end) && is_group_end(total - end))
        .collect::<Vec<_>>();
    let lower = pick_cuts(
        &candidates,
        ((buckets - 1) / 2).min(candidates.len()),
        total,
        buckets,
    );
    lower
        .iter()
        .copied()
        .chain(centre_cut.then_some(half))
        .chain(lower.iter().rev().map(|&cut| total - cut))
        .map(|cut| sorted_pooled[cut - 1])
        .collect()
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

    fn single_sequence(points: &[(f64, f64)]) -> Vec<StoikovObservation> {
        points
            .iter()
            .enumerate()
            .map(|(index, &(mid_price, imbalance))| StoikovObservation {
                sequence_id: 0,
                event_time_us: index as i64,
                mid_price,
                imbalance,
                spread_ticks: 1,
            })
            .collect()
    }

    fn one_spread_config(num_imbalance_buckets: usize, symmetrize: bool) -> StoikovMarkovConfig {
        StoikovMarkovConfig {
            num_imbalance_buckets,
            min_spread_ticks: 1,
            max_spread_ticks: 1,
            symmetrize,
            default_horizon_moves: 6,
        }
    }

    /// Deterministic uniform draws in [0, 1) (64-bit LCG, top 53 bits).
    fn lcg_uniforms(seed: u64) -> impl Iterator<Item = f64> {
        let mut state = seed;
        std::iter::repeat_with(move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
    }

    /// A chain whose state s moves with probability `move_probability[s]`; the remaining
    /// mass of each row, the post-move law and the move sizes are drawn from `seed`.
    fn random_chain(seed: u64, move_probability: &[f64]) -> FirstMoveChain {
        let n = move_probability.len();
        let mut uniforms = lcg_uniforms(seed);
        let mut draw_row = |mass: f64| {
            let weights = uniforms.by_ref().take(n).collect::<Vec<_>>();
            let total = weights.iter().sum::<f64>();
            weights.into_iter().map(move |w| w * mass / total)
        };
        let mut no_move = Vec::with_capacity(n * n);
        let mut after_move = Vec::with_capacity(n * n);
        for &p in move_probability {
            no_move.extend(draw_row(1.0 - p));
            after_move.extend(draw_row(p));
        }
        let expected_move = move_probability
            .iter()
            .zip(lcg_uniforms(seed ^ 0x9e37_79b9))
            .map(|(p, u)| p * (u - 0.5))
            .collect();
        FirstMoveChain {
            state_count: n,
            no_move,
            expected_move,
            after_move,
        }
    }

    /// sum_{k < terms} Q^k [t | R2] by plain power-series summation.
    fn series_reference(chain: &FirstMoveChain, terms: usize) -> (Vec<f64>, Vec<f64>) {
        let n = chain.state_count;
        let mut term_t = chain.expected_move.clone();
        let mut term_r = chain.after_move.clone();
        let mut sum_t = vec![0.0; n];
        let mut sum_r = vec![0.0; n * n];
        for _ in 0..terms {
            sum_t.iter_mut().zip(&term_t).for_each(|(s, t)| *s += t);
            sum_r.iter_mut().zip(&term_r).for_each(|(s, t)| *s += t);
            let mut next_t = vec![0.0; n];
            let mut next_r = vec![0.0; n * n];
            for i in 0..n {
                for k in 0..n {
                    let q = chain.no_move[i * n + k];
                    next_t[i] += q * term_t[k];
                    for j in 0..n {
                        next_r[i * n + j] += q * term_r[k * n + j];
                    }
                }
            }
            term_t = next_t;
            term_r = next_r;
        }
        (sum_t, sum_r)
    }

    fn max_abs_diff<'a>(
        left: impl IntoIterator<Item = &'a f64>,
        right: impl IntoIterator<Item = &'a f64>,
    ) -> f64 {
        left.into_iter()
            .zip(right)
            .map(|(l, r)| (l - r).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn symmetrization_mirrors_values_with_pooled_bounds() {
        // Imbalances are all low. Pooled {I} U {1 - I} = {.1, .2, .2, .3, .7, .8, .8, .9} gives
        // 3-bucket bounds [.2, 1 - .3], so buckets {.1, .2}, {.3, .7}, {.8, .9}. Raw-sample
        // bounds would be [.1, .2] and index mirroring would pool .3 with the mirror of .1.
        // Every step moves the mid, so Q = 0, G1 = t and B = R2.
        let observations =
            single_sequence(&[(100.0, 0.1), (100.5, 0.2), (101.5, 0.3), (101.0, 0.2)]);
        let mut model = StoikovMarkovMicroPrice::new();
        model
            .fit(&observations, 1.0, &one_spread_config(3, true))
            .unwrap();

        assert_eq!(model.imbalance_bounds, vec![0.2, 1.0 - 0.3]);
        // Bucket 0: .1 -> .2 (+.5), .2 -> .3 (+1); bucket 1: .3 -> .2 (-.5) and its mirror
        // .7 -> .8 (+.5); bucket 2: the mirrors .9 -> .8 (-.5), .8 -> .7 (-1).
        let g1 = &model.first_move_adjustments;
        assert!(max_abs_diff(g1, &[0.75, 0.0, -0.75]) < 1e-15, "{g1:?}");
        let expected_b = [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]];
        for (row, expected) in model.transition_after_move.iter().zip(&expected_b) {
            assert!(max_abs_diff(row, expected) < 1e-15, "{row:?}");
        }
    }

    #[test]
    fn symmetrization_bins_mirrors_by_value() {
        // Pooled {.1, .3, .5, .5, .5, .5, .7, .9}: the .5 group straddles the centre, so the
        // symmetric layout of 4 requested buckets falls back to 3 with bounds [.3, .5]
        // (upper-inclusive): {.1, .3}, {.5}, {.7, .9}. The mirror of .5 is .5 itself.
        let observations =
            single_sequence(&[(100.0, 0.1), (100.5, 0.3), (101.0, 0.5), (101.5, 0.5)]);
        let mut model = StoikovMarkovMicroPrice::new();
        model
            .fit(&observations, 1.0, &one_spread_config(4, true))
            .unwrap();

        assert!(max_abs_diff(&model.imbalance_bounds, &[0.3, 0.5]) < 1e-15);
        // Bucket 0: .1 -> .3 (+.5), .3 -> .5 (+.5). Bucket 1: .5 -> .5 (+.5) and its mirror
        // (-.5). Bucket 2: the mirrors .9 -> .7 (-.5), .7 -> .5 (-.5).
        let g1 = &model.first_move_adjustments;
        assert!(max_abs_diff(g1, &[0.5, 0.0, -0.5]) < 1e-15, "{g1:?}");
        let expected_b = [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]];
        for (row, expected) in model.transition_after_move.iter().zip(&expected_b) {
            assert!(max_abs_diff(row, expected) < 1e-15, "{row:?}");
        }
    }

    #[test]
    fn symmetrized_fit_stays_antisymmetric_with_exact_half_imbalances() {
        // Lot-quantized books often have bid_qty == ask_qty. With an even bucket count the
        // self-mirrored 0.5 group straddles the centre; the layout must still be symmetric
        // or G1 loses antisymmetry and G* drifts.
        let mut uniforms = lcg_uniforms(5);
        let mut mid = 100.0;
        let observations = (0..20_000)
            .map(|index| {
                let imbalance = match uniforms.next().unwrap() {
                    u if u < 0.05 => 0.5,
                    u => u * u,
                };
                let u = uniforms.next().unwrap();
                mid += if u < 0.2 * imbalance {
                    0.5
                } else if u > 1.0 - 0.2 * (1.0 - imbalance) {
                    -0.5
                } else {
                    0.0
                };
                StoikovObservation {
                    sequence_id: index / 5_000,
                    event_time_us: index as i64,
                    mid_price: mid,
                    imbalance,
                    spread_ticks: 1,
                }
            })
            .collect::<Vec<_>>();
        let mut model = StoikovMarkovMicroPrice::new();
        model
            .fit(&observations, 1.0, &one_spread_config(10, true))
            .unwrap();

        assert_eq!(model.bucket_count(), 9);
        assert_eq!(model.state_index(0.5, 1), Some(4));
        let n = model.bucket_count();
        let g1 = &model.first_move_adjustments;
        let b = &model.transition_after_move;
        for k in 0..n {
            assert!((g1[k] + g1[n - 1 - k]).abs() < 1e-15, "{g1:?}");
            for j in 0..n {
                assert!((b[k][j] - b[n - 1 - k][n - 1 - j]).abs() < 1e-15);
            }
        }
        let limit = model.limit_adjustments(1e-12, 1_000_000);
        assert!(limit.is_converged(), "{limit:?}");
        assert!(limit.moves() < 1_000, "{limit:?}");
    }

    #[test]
    fn mirrored_float_ties_share_a_bucket() {
        // 1 - 0.8 = 0.19999999999999996 != 0.2: without tie merging the pooled sample has
        // three groups and the extra bucket is a closed non-moving state.
        let observations = single_sequence(&[(100.0, 0.2), (100.5, 0.8), (100.5, 0.8)]);
        let mut model = StoikovMarkovMicroPrice::new();
        model
            .fit(&observations, 1.0, &one_spread_config(10, true))
            .unwrap();
        assert_eq!(model.imbalance_bounds, vec![0.2]);
        assert_eq!(model.first_move_adjustments, vec![0.5, -0.5]);
    }

    #[test]
    fn mirror_merge_matches_the_sorted_pooled_sample() {
        // Ties, self-mirrored halves, the ends and values whose mirror rounds.
        let imbalances = lcg_uniforms(3)
            .take(5_000)
            .map(|u| match (u * 1e4) as u32 % 7 {
                0 => 0.5,
                1 => 0.0,
                2 => 1.0,
                3 => 0.8,
                4 => 0.2,
                _ => u * u,
            })
            .collect::<Vec<_>>();
        let mut pooled = imbalances
            .iter()
            .flat_map(|&i| [i, 1.0 - i])
            .collect::<Vec<_>>();
        pooled.sort_unstable_by(f64::total_cmp);
        let mut sorted = imbalances;
        sorted.sort_unstable_by(f64::total_cmp);
        let merged = merge_with_mirror(&sorted);
        assert_eq!(merged.len(), pooled.len());
        assert!(
            merged
                .iter()
                .zip(&pooled)
                .all(|(m, p)| m.to_bits() == p.to_bits())
        );
    }

    #[test]
    fn symmetrized_bounds_are_mirror_images() {
        // Skewed continuous sample I = u^2; mirrored bounds differ only by one sample because
        // buckets are upper-inclusive.
        let imbalances = lcg_uniforms(7)
            .take(4_000)
            .map(|u| u * u)
            .collect::<Vec<_>>();
        let observations = imbalances
            .iter()
            .enumerate()
            .map(|(index, &imbalance)| StoikovObservation {
                sequence_id: 0,
                event_time_us: index as i64,
                mid_price: 100.0 + 0.5 * (index % 3) as f64,
                imbalance,
                spread_ticks: 1,
            })
            .collect::<Vec<_>>();
        let mut model = StoikovMarkovMicroPrice::new();
        model
            .fit(&observations, 1.0, &one_spread_config(10, true))
            .unwrap();

        let mut pooled = imbalances
            .iter()
            .flat_map(|&i| [i, 1.0 - i])
            .collect::<Vec<_>>();
        pooled.sort_unstable_by(f64::total_cmp);
        let bounds = &model.imbalance_bounds;
        assert_eq!(bounds.len(), 9);
        for (bound, mirror) in bounds.iter().zip(bounds.iter().rev()) {
            let reflected = 1.0 - mirror;
            let (lo, hi) = (bound.min(reflected), bound.max(reflected));
            assert!(hi - lo < 1e-3, "{bound} vs {reflected}");
            let strictly_between = pooled
                .iter()
                .filter(|&&x| x > lo + 1e-15 && x < hi - 1e-15)
                .count();
            assert_eq!(strictly_between, 0, "{bound} vs {reflected}");
        }
    }

    #[test]
    fn direct_solve_matches_the_infinite_series() {
        let fast = random_chain(11, &[0.3, 0.5, 0.2, 0.8, 0.4]);
        // Moves once per ~1000 steps: 256 series terms keep only ~1 - 0.999^256 of G1.
        let slow = random_chain(23, &[1e-3, 2e-3, 1.5e-3, 1e-3]);
        for (chain, terms) in [(&fast, 2_000), (&slow, 60_000)] {
            let (g1, b) = chain.solve().unwrap();
            let (ref_g1, ref_b) = series_reference(chain, terms);
            assert!(max_abs_diff(&g1, &ref_g1) < 1e-10, "{g1:?} vs {ref_g1:?}");
            assert!(max_abs_diff(b.iter().flatten(), &ref_b) < 1e-10);
            // B rows are the law of the post-move state: they sum to one.
            for row in &b {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-10, "{row:?}");
            }
        }

        let (g1, _) = slow.solve().unwrap();
        let (truncated, _) = series_reference(&slow, 256);
        let ratio = truncated
            .iter()
            .zip(&g1)
            .map(|(t, g)| t / g)
            .fold(f64::INFINITY, f64::min);
        assert!(ratio < 0.5, "256 terms kept {ratio} of G1");
    }

    #[test]
    fn closed_non_moving_states_are_rejected() {
        // States 0 and 1 swap forever without moving the mid; state 2 moves.
        let chain = FirstMoveChain {
            state_count: 3,
            no_move: vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            expected_move: vec![0.0, 0.0, 0.25],
            after_move: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        };
        assert!(matches!(
            chain.solve(),
            Err(StoikovMarkovError::NonMovingClosedClass { .. })
        ));

        // Through `fit`: a single state that only ever stays put.
        let observations = single_sequence(&[(100.0, 0.5), (100.0, 0.5), (100.0, 0.5)]);
        let error = StoikovMarkovMicroPrice::new()
            .fit(&observations, 1.0, &one_spread_config(1, false))
            .unwrap_err();
        assert!(matches!(
            error,
            StoikovMarkovError::NonMovingClosedClass { state: 0 }
        ));
    }

    #[test]
    fn limit_adjustments_reach_the_limit_of_a_mixing_chain() {
        // Low imbalance always ticks down and lands in either bucket with equal odds; the
        // mirrored high-imbalance state ticks up. B mixes to the stationary law, so
        // B * G1 = 0 and G* equals G1.
        let observations = single_sequence(&[(100.0, 0.1), (99.5, 0.1), (99.0, 0.9)]);
        let mut model = StoikovMarkovMicroPrice::new();
        model
            .fit(&observations, 1.0, &one_spread_config(2, true))
            .unwrap();

        let limit = model.limit_adjustments(1e-12, 1_000);
        assert!(limit.is_converged(), "{limit:?}");
        assert_eq!(limit.moves(), 1);
        assert_eq!(limit.adjustments(), &[-0.5, 0.5]);
        assert_eq!(
            limit.adjustments(),
            model.adjustments_for_horizon(limit.moves())
        );

        // max_moves = 1 still checks B * G1.
        assert_eq!(
            model.limit_adjustments(1e-12, 1),
            MoveSeriesLimit::Converged {
                adjustments: vec![-0.5, 0.5],
                moves: 1
            }
        );
        // max_moves = 0 sums nothing; it converged only if G1 itself is within tolerance.
        assert_eq!(
            model.limit_adjustments(1e-12, 0),
            MoveSeriesLimit::Capped {
                adjustments: vec![0.0, 0.0],
                moves: 0
            }
        );
        assert_eq!(
            model.limit_adjustments(0.5, 0),
            MoveSeriesLimit::Converged {
                adjustments: vec![0.0, 0.0],
                moves: 0
            }
        );
        assert_eq!(model.adjustments_for_horizon(0), vec![0.0, 0.0]);
    }

    #[test]
    fn limit_adjustments_cap_a_drifting_chain() {
        // Unsymmetrized chain where the only state keeps ticking down: G_k = -k / 2.
        let observations = single_sequence(&[(100.0, 0.1), (99.5, 0.1), (99.0, 0.1)]);
        let mut model = StoikovMarkovMicroPrice::new();
        model
            .fit(&observations, 1.0, &one_spread_config(1, false))
            .unwrap();

        let limit = model.limit_adjustments(1e-12, 50);
        assert!(!limit.is_converged(), "{limit:?}");
        assert_eq!(limit.moves(), 50);
        assert!((limit.adjustments()[0] + 25.0).abs() < 1e-9, "{limit:?}");
        assert_eq!(limit.adjustments(), model.adjustments_for_horizon(50));
    }

    #[test]
    fn limit_adjustments_flag_a_drifting_chain_that_leaks() {
        // Every move ticks down; the last observation's bucket is never seen as current, so
        // B = [[198/199, 1/199], [0, 0]] and B^k G1 decays only because mass leaks.
        let points = (0..200)
            .map(|index| {
                (
                    100.0 - 0.5 * index as f64,
                    if index < 199 { 0.1 } else { 0.9 },
                )
            })
            .collect::<Vec<_>>();
        let mut model = StoikovMarkovMicroPrice::new();
        model
            .fit(&single_sequence(&points), 1.0, &one_spread_config(2, false))
            .unwrap();
        assert_eq!(model.first_move_adjustments, vec![-0.5, 0.0]);

        let limit = model.limit_adjustments(1e-12, 1_000_000);
        let MoveSeriesLimit::Leaked { adjustments, moves } = &limit else {
            panic!("{limit:?}");
        };
        assert!(!limit.is_converged());
        // (198/199)^k <= EXHAUSTED_MASS first at k = ceil(ln 1e-12 / ln(198/199)).
        let expected_moves = (EXHAUSTED_MASS.ln() / (198.0f64 / 199.0).ln()).ceil() as usize;
        assert!(
            moves.abs_diff(expected_moves) <= 1,
            "{moves} vs {expected_moves}"
        );
        assert_eq!(*adjustments, model.adjustments_for_horizon(*moves));
    }

    #[test]
    fn limit_adjustments_converge_despite_a_leaking_state() {
        // Symmetrized, 3 buckets at [.1, .5]. The middle bucket only ever ends the sequence,
        // so it has a zero B row. On buckets {0, 2}, B = [[.25, .5], [.5, .25]] and G1 is
        // antisymmetric, so B^k G1 = (-1/4)^k G1 while the surviving mass is (3/4)^k:
        // G* = G1 / (1 + 1/4).
        let observations = single_sequence(&[
            (100.0, 0.1),
            (99.5, 0.1),
            (99.0, 0.9),
            (99.5, 0.1),
            (99.0, 0.5),
        ]);
        let mut model = StoikovMarkovMicroPrice::new();
        model
            .fit(&observations, 1.0, &one_spread_config(3, true))
            .unwrap();
        assert_eq!(model.transition_after_move[1], vec![0.0; 3]);

        let limit = model.limit_adjustments(1e-12, 1_000);
        assert!(limit.is_converged(), "{limit:?}");
        assert!(max_abs_diff(limit.adjustments(), &[-0.4, 0.0, 0.4]) < 1e-12);
        assert_eq!(
            limit.adjustments(),
            model.adjustments_for_horizon(limit.moves())
        );
    }

    #[test]
    fn limit_adjustments_of_an_empty_model_converge() {
        assert_eq!(
            StoikovMarkovMicroPrice::new().limit_adjustments(1e-12, 10),
            MoveSeriesLimit::Converged {
                adjustments: Vec::new(),
                moves: 0
            }
        );
    }

    #[test]
    fn limit_adjustments_never_converge_on_nan() {
        let observations = single_sequence(&[(100.0, 0.1), (99.5, 0.1), (99.0, 0.9)]);
        let mut model = StoikovMarkovMicroPrice::new();
        model
            .fit(&observations, 1.0, &one_spread_config(2, true))
            .unwrap();
        model.first_move_adjustments[0] = f64::NAN;
        let limit = model.limit_adjustments(1e-12, 10);
        assert!(
            matches!(limit, MoveSeriesLimit::Capped { moves: 10, .. }),
            "{limit:?}"
        );
    }

    #[test]
    fn loads_models_saved_with_series_fields() {
        let path = temp_model_path("legacy_series_fields");
        fs::write(
            &path,
            r#"{"imbalance_bounds":[0.5],"min_spread_ticks":1,"max_spread_ticks":1,
                "price_step":0.5,"default_horizon_moves":6,"symmetrized":true,
                "first_move_adjustments":[-0.5,0.5],
                "transition_after_move":[[0.5,0.5],[0.5,0.5]],
                "default_adjustments":[-0.5,0.5],"series_terms_used":256}"#,
        )
        .unwrap();
        let model = StoikovMarkovMicroPrice::load_model(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(model.first_move_adjustments, vec![-0.5, 0.5]);
        assert_eq!(model.get_adjustment(0.9, 1), 0.5);

        let config: StoikovMarkovConfig = serde_json::from_str(
            r#"{"num_imbalance_buckets":10,"min_spread_ticks":1,"max_spread_ticks":6,
                "symmetrize":true,"default_horizon_moves":6,"max_series_terms":256,
                "convergence_tol":1e-12}"#,
        )
        .unwrap();
        assert_eq!(config, StoikovMarkovConfig::default());
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
