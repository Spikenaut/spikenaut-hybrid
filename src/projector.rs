//! Spike-to-expert projector — the heart of the neuromorphic-ANN fusion.
//!
//! The [`Projector`] sits between the Spikenaut SNN and OLMoE's expert router.
//! Its job is to compress the high-dimensional spike activity (neuron indices +
//! firing rates + membrane potentials) into a single dense embedding vector
//! that OLMoE can use as a context prefix.
//!
//! # Projection modes
//!
//! | Mode | Description | Best for |
//! |------|-------------|----------|
//! | [`RateSum`] | Per-neuron spike-count → linear projection | rate-coded telemetry |
//! | [`TemporalHistogram`] | Spikes binned over time → flatten | timing-sensitive HFT |
//! | [`MembraneSnapshot`] | Post-step membrane potentials → linear | smooth gradient flow |
//!
//! # No OLMoE dependency
//!
//! The projector is intentionally **pure** — it depends only on the spike
//! activity types produced by `neuromod` and emits a plain `Vec<f32>`.
//! This keeps it reusable with any LLM backend.
//!
//! [`RateSum`]: crate::types::ProjectionMode::RateSum
//! [`TemporalHistogram`]: crate::types::ProjectionMode::TemporalHistogram
//! [`MembraneSnapshot`]: crate::types::ProjectionMode::MembraneSnapshot

use crate::error::{HybridError, Result};
use crate::types::{ProjectionMode, EMBEDDING_DIM, SNN_INPUT_CHANNELS};

// ── Projection weight matrix ───────────────────────────────────────────────────

/// Number of neurons in the Spikenaut SNN.
const SNN_NEURONS: usize = 16;

/// Number of Izhikevich neurons in the adaptive bank.
const IZ_NEURONS: usize = 5;

/// Feature vector length before the linear projection.
/// = rate features (16) + temporal bins (16 × 4) + membrane (16) + Iz potentials (5)
const FEATURE_DIM: usize = SNN_NEURONS + (SNN_NEURONS * 4) + SNN_NEURONS + IZ_NEURONS;

// ── Projector ─────────────────────────────────────────────────────────────────

/// Converts Spikenaut SNN output into a dense embedding for OLMoE.
///
/// Internally this is a **learned linear layer** `W ∈ ℝ^{EMBEDDING_DIM × FEATURE_DIM}`
/// plus a bias `b ∈ ℝ^{EMBEDDING_DIM}`, initialised with Xavier-uniform weights.
/// The weight matrix is updated only by Julia/E-prop via the spine; it is
/// **frozen from the Rust side**.
///
/// ```no_run
/// use spikenaut_hybrid::projector::Projector;
/// use spikenaut_hybrid::types::ProjectionMode;
///
/// let proj = Projector::new(ProjectionMode::RateSum);
/// ```
pub struct Projector {
    /// Projection strategy.
    mode: ProjectionMode,

    /// Flat weight matrix `W`, row-major layout: `W[out * FEATURE_DIM + in]`.
    /// Shape: `[EMBEDDING_DIM, FEATURE_DIM]`.
    weights: Vec<f32>,

    /// Bias vector `b`.  Shape: `[EMBEDDING_DIM]`.
    bias: Vec<f32>,

    /// Running exponential moving average of firing rates (for normalisation).
    /// Updated each call to [`project`](Self::project).
    rate_ema: [f32; SNN_NEURONS],

    /// EMA decay constant for firing rate normalisation.
    ema_alpha: f32,
}

impl Projector {
    /// Create a new Projector with Xavier-uniform initialised weights.
    ///
    /// # Arguments
    /// * `mode` — how to aggregate spike activity into a feature vector.
    pub fn new(mode: ProjectionMode) -> Self {
        let fan_in = FEATURE_DIM as f32;
        let fan_out = EMBEDDING_DIM as f32;
        let limit = (6.0_f32 / (fan_in + fan_out)).sqrt();

        // Deterministic Xavier-uniform init (no external rng dep needed).
        let mut weights = Vec::with_capacity(EMBEDDING_DIM * FEATURE_DIM);
        for i in 0..(EMBEDDING_DIM * FEATURE_DIM) {
            // Simple deterministic pseudo-random from index hash.
            let t = ((i as f32 * 1.6180339887) % 1.0) * 2.0 - 1.0;
            weights.push(t * limit);
        }

        Self {
            mode,
            weights,
            bias: vec![0.0; EMBEDDING_DIM],
            rate_ema: [0.0; SNN_NEURONS],
            ema_alpha: 0.1,
        }
    }

    /// Project SNN spike activity into a dense embedding.
    ///
    /// # Arguments
    /// * `spike_train`  — per-step spike sets from `SpikingNetwork::step`.
    /// * `potentials`   — membrane potentials after the final SNN time-step.
    /// * `iz_potentials`— Izhikevich neuron voltages (5 adaptive neurons).
    ///
    /// # Returns
    /// Dense embedding `Vec<f32>` of length [`EMBEDDING_DIM`].
    pub fn project(
        &mut self,
        spike_train: &[Vec<usize>],
        potentials: &[f32],
        iz_potentials: &[f32],
    ) -> Result<Vec<f32>> {
        if potentials.len() < SNN_NEURONS {
            return Err(HybridError::InputLengthMismatch {
                expected: SNN_NEURONS,
                got: potentials.len(),
            });
        }

        let feature_vec = self.build_feature_vector(spike_train, potentials, iz_potentials);
        Ok(self.linear_project(&feature_vec))
    }

    // ── Feature construction ──────────────────────────────────────────────────

    fn build_feature_vector(
        &mut self,
        spike_train: &[Vec<usize>],
        potentials: &[f32],
        iz_potentials: &[f32],
    ) -> Vec<f32> {
        let n_steps = spike_train.len().max(1) as f32;

        // 1. Firing rates per neuron [16 dims]
        let mut rates = [0.0_f32; SNN_NEURONS];
        for step in spike_train {
            for &idx in step {
                if idx < SNN_NEURONS {
                    rates[idx] += 1.0;
                }
            }
        }
        for r in &mut rates {
            *r /= n_steps;
        }

        // Update EMA for normalisation
        for i in 0..SNN_NEURONS {
            self.rate_ema[i] =
                self.ema_alpha * rates[i] + (1.0 - self.ema_alpha) * self.rate_ema[i];
        }

        // 2. Temporal histogram bins (4 equal-width bins) [64 dims]
        let bins = 4usize;
        let mut hist = vec![0.0_f32; SNN_NEURONS * bins];
        if !spike_train.is_empty() {
            let steps = spike_train.len();
            for (t, step) in spike_train.iter().enumerate() {
                let bin = ((t * bins) / steps).min(bins - 1);
                for &idx in step {
                    if idx < SNN_NEURONS {
                        hist[idx * bins + bin] += 1.0;
                    }
                }
            }
            let total = n_steps / bins as f32;
            for h in &mut hist {
                *h /= total.max(1.0);
            }
        }

        // 3. Membrane potentials [16 dims] — clamped to [0, 1]
        let membrane: Vec<f32> = potentials[..SNN_NEURONS]
            .iter()
            .map(|&v| v.clamp(0.0, 1.0))
            .collect();

        // 4. Izhikevich adaptive bank potentials [5 dims]
        let iz: Vec<f32> = iz_potentials
            .iter()
            .take(IZ_NEURONS)
            .map(|&v| (v / 30.0).clamp(-1.0, 1.0)) // Izhikevich Vpeak ≈ 30 mV
            .chain(std::iter::repeat(0.0))
            .take(IZ_NEURONS)
            .collect();

        // Mode-specific blending
        let mut features = Vec::with_capacity(FEATURE_DIM);
        match self.mode {
            ProjectionMode::RateSum => {
                features.extend_from_slice(&rates);
                features.extend_from_slice(&hist);
                features.extend_from_slice(&membrane);
                features.extend_from_slice(&iz);
            }
            ProjectionMode::TemporalHistogram => {
                // Weight histogram more heavily than raw rates
                let weighted_rates: Vec<f32> = rates.iter().map(|r| r * 0.3).collect();
                features.extend_from_slice(&weighted_rates);
                let weighted_hist: Vec<f32> = hist.iter().map(|h| h * 2.0).collect();
                features.extend_from_slice(&weighted_hist);
                features.extend_from_slice(&membrane);
                features.extend_from_slice(&iz);
            }
            ProjectionMode::MembraneSnapshot => {
                // Use membrane directly as primary signal
                let membrane_primary: Vec<f32> =
                    membrane.iter().map(|v| v * 2.0).collect();
                features.extend_from_slice(&rates);
                features.extend_from_slice(&hist);
                features.extend_from_slice(&membrane_primary);
                features.extend_from_slice(&iz);
            }
        }

        // Pad or truncate to exactly FEATURE_DIM
        features.resize(FEATURE_DIM, 0.0);
        features
    }

    // ── Linear projection W × f + b ──────────────────────────────────────────

    fn linear_project(&self, features: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; EMBEDDING_DIM];
        for out_i in 0..EMBEDDING_DIM {
            let mut acc = self.bias[out_i];
            let row_offset = out_i * FEATURE_DIM;
            for in_j in 0..features.len().min(FEATURE_DIM) {
                acc += self.weights[row_offset + in_j] * features[in_j];
            }
            // Layer norm approximation: tanh squash keeps embedding bounded
            out[out_i] = acc.tanh();
        }
        out
    }

    // ── Weight management (for spine / E-prop updates) ────────────────────────

    /// Replace the weight matrix with values received from `SpikenautDistill.jl`.
    ///
    /// Julia sends the updated projector weights as a flat `f32` slice via the
    /// spine after each E-prop step.
    ///
    /// # Errors
    /// Returns [`HybridError::InputLengthMismatch`] if the slice length ≠
    /// `EMBEDDING_DIM × FEATURE_DIM`.
    pub fn load_weights(&mut self, weights: &[f32]) -> Result<()> {
        let expected = EMBEDDING_DIM * FEATURE_DIM;
        if weights.len() != expected {
            return Err(HybridError::InputLengthMismatch {
                expected,
                got: weights.len(),
            });
        }
        self.weights.copy_from_slice(weights);
        Ok(())
    }

    /// Replace the bias vector.
    pub fn load_bias(&mut self, bias: &[f32]) -> Result<()> {
        if bias.len() != EMBEDDING_DIM {
            return Err(HybridError::InputLengthMismatch {
                expected: EMBEDDING_DIM,
                got: bias.len(),
            });
        }
        self.bias.copy_from_slice(bias);
        Ok(())
    }

    /// Current projection mode.
    pub fn mode(&self) -> ProjectionMode {
        self.mode
    }

    /// Dimensionality constants (useful for allocating buffers).
    pub fn dims(&self) -> (usize, usize) {
        (FEATURE_DIM, EMBEDDING_DIM)
    }

    /// Firing rate EMA snapshot (useful for diagnostics / reward shaping).
    pub fn rate_ema(&self) -> &[f32; SNN_NEURONS] {
        &self.rate_ema
    }
}

impl Default for Projector {
    fn default() -> Self {
        Self::new(ProjectionMode::RateSum)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_spike_train(n_steps: usize) -> Vec<Vec<usize>> {
        (0..n_steps)
            .map(|t| vec![t % SNN_NEURONS, (t + 1) % SNN_NEURONS])
            .collect()
    }

    #[test]
    fn test_project_output_length() {
        let mut proj = Projector::new(ProjectionMode::RateSum);
        let spikes = dummy_spike_train(20);
        let potentials = vec![0.3; SNN_NEURONS];
        let iz_pots = vec![15.0; IZ_NEURONS];
        let embedding = proj.project(&spikes, &potentials, &iz_pots).unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_project_values_bounded() {
        let mut proj = Projector::new(ProjectionMode::TemporalHistogram);
        let spikes = dummy_spike_train(10);
        let potentials = vec![0.5; SNN_NEURONS];
        let iz_pots = vec![30.0; IZ_NEURONS];
        let embedding = proj.project(&spikes, &potentials, &iz_pots).unwrap();
        for v in &embedding {
            assert!(
                v.abs() <= 1.0,
                "embedding value {v} out of tanh range [-1, 1]"
            );
        }
    }

    #[test]
    fn test_load_weights_length_check() {
        let mut proj = Projector::new(ProjectionMode::RateSum);
        let bad_weights = vec![0.0f32; 10]; // wrong length
        assert!(proj.load_weights(&bad_weights).is_err());
    }

    #[test]
    fn test_membrane_mode() {
        let mut proj = Projector::new(ProjectionMode::MembraneSnapshot);
        let spikes = dummy_spike_train(5);
        let potentials = vec![0.8; SNN_NEURONS];
        let iz_pots = vec![0.0; IZ_NEURONS];
        let embedding = proj.project(&spikes, &potentials, &iz_pots).unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_dims() {
        let proj = Projector::default();
        let (feat, emb) = proj.dims();
        assert_eq!(feat, FEATURE_DIM);
        assert_eq!(emb, EMBEDDING_DIM);
    }
}
