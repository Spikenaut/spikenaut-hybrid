//! `HybridModel` — the top-level orchestrator for the SpikeLMo architecture.
//!
//! This is the single struct researchers interact with.  It wires together all
//! five stages of the fusion pipeline:
//!
//! ```text
//! TelemetrySnapshot
//!       │
//!       ▼ spikenaut-encoder  (NeuromodSensoryEncoder)
//! [f32; 8] telemetry stimuli → [f32; 16] Poisson rates
//!       │
//!       ▼ neuromod  (SpikingNetwork × snn_steps)
//! spike_train: Vec<Vec<usize>>  +  membrane_potentials: Vec<f32>
//!       │
//!       ▼ spikenaut-hybrid  (Projector)
//! embedding: Vec<f32>  [EMBEDDING_DIM = 2048]
//!       │
//!       ▼ spikenaut-hybrid  (OLMoE — frozen)
//! OlmoeOutput { expert_weights, selected_experts, hidden }
//!       │
//!       ▼ spikenaut-spine  (optional, feature = spine-zmq)
//! TrainSignal ──► SpikenautDistill.jl  (E-prop on SNN only)
//! ```

use crate::error::{HybridError, Result};
use crate::olmoe::OLMoE;
use crate::projector::Projector;
use crate::types::{
    EMBEDDING_DIM, HybridConfig, HybridOutput, SNN_INPUT_CHANNELS, TrainSignal,
};

use neuromod::SpikingNetwork;
use spikenaut_encoder::encoders::neuromod::NeuromodSensoryEncoder;
use spikenaut_encoder::modulators::NeuroModulators as EncoderNeuroModulators;
use spikenaut_encoder::Encoder;
use spikenaut_telemetry::TelemetrySnapshot;

// ── HybridModel ───────────────────────────────────────────────────────────────

/// Neuromorphic-ANN hybrid model for the SpikeLMo framework.
///
/// Combines:
/// - **Spikenaut SNN** (`neuromod::SpikingNetwork`) — event-driven, learning
///   via E-prop/OTTT from `SpikenautDistill.jl`.
/// - **`spikenaut-encoder`** (`NeuromodSensoryEncoder`) — converts hardware
///   telemetry / HFT market data into 16-channel neuromodulated spike stimuli.
/// - **`Projector`** — projects SNN spike activity into a dense OLMoE embedding.
/// - **`OLMoE`** — frozen Mixture-of-Experts LLM providing contextual reasoning.
/// - **`spikenaut-spine`** — (optional) ZMQ bridge to Julia for SNN training.
///
/// # Quick start
/// ```no_run
/// use spikenaut_hybrid::{HybridConfig, HybridModel};
/// use spikenaut_telemetry::TelemetrySnapshot;
///
/// let cfg = HybridConfig::default();
/// let mut model = HybridModel::new(cfg).unwrap();
///
/// let snap = TelemetrySnapshot::default();
/// let output = model.forward(&snap).unwrap();
///
/// println!("Fired neurons: {:?}", output.spike_train.last());
/// println!("Expert selected: {:?}", output.selected_experts);
/// ```
pub struct HybridModel {
    config: HybridConfig,
    encoder: NeuromodSensoryEncoder,
    snn: SpikingNetwork,
    projector: Projector,
    olmoe: OLMoE,
    global_step: i64,
}

impl HybridModel {
    /// Build a new `HybridModel` from a [`HybridConfig`].
    ///
    /// # Errors
    /// * [`HybridError::InvalidConfig`] — bad config values.
    /// * [`HybridError::ModelLoad`] — GGUF/safetensors file unreadable.
    pub fn new(config: HybridConfig) -> Result<Self> {
        if config.snn_steps == 0 {
            return Err(HybridError::InvalidConfig("snn_steps must be ≥ 1".into()));
        }
        if config.context_length == 0 {
            return Err(HybridError::InvalidConfig("context_length must be ≥ 1".into()));
        }
        if config.top_k_experts > config.num_experts {
            return Err(HybridError::InvalidConfig(format!(
                "top_k_experts ({}) > num_experts ({})",
                config.top_k_experts, config.num_experts
            )));
        }

        let mut encoder = NeuromodSensoryEncoder::new(8, SNN_INPUT_CHANNELS);
        encoder.update_neuromodulators(EncoderNeuroModulators {
            mining_dopamine: config.initial_dopamine - 0.5,
            ..Default::default()
        });

        let snn = SpikingNetwork::new();
        let projector = Projector::new(config.projection_mode);
        let olmoe = OLMoE::load_with_mode(
            &config.olmoe_model_path,
            config.num_experts,
            config.top_k_experts,
            config.olmoe_execution_mode,
        )?;

        Ok(Self { config, encoder, snn, projector, olmoe, global_step: 0 })
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /// Run a full forward pass through the hybrid pipeline.
    ///
    /// # Pipeline stages
    /// 1. Convert snapshot → 8-channel stimuli.
    /// 2. Update encoder neuromodulators.
    /// 3. Encode to 16-channel Poisson rates → step SNN `snn_steps` times.
    /// 4. Project spike activity → dense embedding.
    /// 5. Run frozen OLMoE → expert selection.
    /// 6. Assemble [`HybridOutput`].
    pub fn forward(&mut self, snap: &TelemetrySnapshot) -> Result<HybridOutput> {
        self.global_step += 1;

        // Stage 1 & 2
        let market_inputs = self.snapshot_to_stimuli(snap);
        self.update_neuromodulators_from_snapshot(snap);

        // Stage 3: encode + SNN
        let enc_mods = self.encoder_modulators_from_snapshot(snap);
        self.encoder.update_neuromodulators(enc_mods);
        let encoded = self.encoder.encode(&market_inputs);
        let stimuli_16 = Self::encoded_output_to_stimuli(&encoded);
        let neuromod = bridge_modulators(&enc_mods);

        let mut spike_train: Vec<Vec<usize>> = Vec::with_capacity(self.config.snn_steps);
        for _ in 0..self.config.snn_steps {
            spike_train.push(self.snn.step(&stimuli_16, &neuromod));
        }

        // Stage 4: project
        let potentials = self.snn.get_membrane_potentials();
        let iz_potentials: Vec<f32> = self.snn.iz_neurons.iter().map(|iz| iz.v).collect();
        let embedding = self.projector.project(&spike_train, &potentials, &iz_potentials)?;

        // Stage 5: OLMoE
        let olmoe_out = self.olmoe.forward(&embedding)?;

        // Stage 6: assemble
        let n_neurons = potentials.len();
        let steps_f = self.config.snn_steps as f32;
        let mut counts = vec![0usize; n_neurons];
        for step in &spike_train {
            for &idx in step {
                if idx < n_neurons { counts[idx] += 1; }
            }
        }
        let firing_rates: Vec<f32> = counts.iter().map(|&c| c as f32 / steps_f).collect();

        Ok(HybridOutput {
            spike_train,
            firing_rates,
            membrane_potentials: potentials,
            embedding,
            expert_weights: Some(olmoe_out.expert_weights),
            selected_experts: Some(olmoe_out.selected_experts),
            reasoning: None,
        })
    }

    // ── Training hook ─────────────────────────────────────────────────────────

    /// Forward pass + MSE loss + E-prop signal to Julia via the spine.
    ///
    /// OLMoE weights are **never** updated here — only the SNN learns.
    ///
    /// # Arguments
    /// * `target` — regression target, length `EMBEDDING_DIM`.
    ///
    /// # Returns
    /// MSE loss between the SNN→projector embedding and the target.
    pub fn train_step(&mut self, snap: &TelemetrySnapshot, target: &[f32]) -> Result<f32> {
        if target.len() != EMBEDDING_DIM {
            return Err(HybridError::InputLengthMismatch {
                expected: EMBEDDING_DIM,
                got: target.len(),
            });
        }

        let output = self.forward(snap)?;

        let loss: f32 = output
            .embedding
            .iter()
            .zip(target.iter())
            .map(|(h, t)| (h - t).powi(2))
            .sum::<f32>()
            / EMBEDDING_DIM as f32;

        let eligibility_traces: Vec<f32> = self.projector.rate_ema().iter().copied().collect();

        let signal = TrainSignal {
            timestamp_ms: snap.timestamp_ms,
            prediction_loss: loss,
            thermal_penalty: snap.thermal_stress(),
            eligibility_traces,
            global_step: self.global_step,
        };

        self.publish_train_signal(&signal)?;
        Ok(loss)
    }

    // ── Reset ─────────────────────────────────────────────────────────────────

    /// Reset SNN state to initial conditions (membrane potentials, spike times,
    /// STDP eligibility traces).  Projector weights are unaffected.
    /// Also resets the projector and OLMoE GIF membrane state so no stale
    /// spiking history bleeds across episodes.
    pub fn reset(&mut self) {
        self.snn.reset();
        self.projector.reset_membrane();
        self.olmoe.reset_state();
        self.global_step = 0;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Current global step counter.
    pub fn global_step(&self) -> i64 { self.global_step }

    /// `true` if OLMoE was loaded from disk; `false` = stub mode.
    pub fn olmoe_loaded(&self) -> bool { self.olmoe.is_loaded() }

    /// Active configuration.
    pub fn config(&self) -> &HybridConfig { &self.config }

    /// Mutable projector access (e.g. to load new weights from Julia).
    pub fn projector_mut(&mut self) -> &mut Projector { &mut self.projector }

    /// Immutable SNN access (for diagnostics / analysis).
    pub fn snn(&self) -> &SpikingNetwork { &self.snn }

    /// Mutable SNN access (e.g. for manual reward injection).
    pub fn snn_mut(&mut self) -> &mut SpikingNetwork { &mut self.snn }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Map a [`TelemetrySnapshot`] to 8 normalised `[0, 1]` stimuli channels.
    ///
    /// | Ch | Signal                          |
    /// |----|---------------------------------|
    /// | 0  | GPU temperature stress          |
    /// | 1  | GPU power load                  |
    /// | 2  | CPU temperature stress          |
    /// | 3  | CPU package power               |
    /// | 4  | Mining efficiency (dopamine)    |
    /// | 5  | Dynex hashrate (normalised)     |
    /// | 6  | Qubic tick activity             |
    /// | 7  | Ocean intelligence signal       |
    fn snapshot_to_stimuli(&self, snap: &TelemetrySnapshot) -> [f32; 8] {
        [
            ((snap.gpu_temp_c - 60.0) / 30.0).clamp(0.0, 1.0),
            (snap.gpu_power_w / 450.0).clamp(0.0, 1.0),
            ((snap.cpu_tctl_c - 40.0) / 50.0).clamp(0.0, 1.0),
            (snap.cpu_package_power_w / 150.0).clamp(0.0, 1.0),
            snap.workload_efficiency as f32,
            (snap.workload_throughput as f32 / 1000.0).clamp(0.0, 1.0),
            (snap.mem_util_pct / 100.0).clamp(0.0, 1.0),
            ((snap.auxiliary_signal as f32).abs()).clamp(0.0, 1.0),
        ]
    }

    /// Push live telemetry signals into the encoder's neuromodulator state.
    fn update_neuromodulators_from_snapshot(&mut self, snap: &TelemetrySnapshot) {
        let modulators = self.encoder_modulators_from_snapshot(snap);
        self.encoder.update_neuromodulators(modulators);
    }

    fn encoder_modulators_from_snapshot(&self, snap: &TelemetrySnapshot) -> EncoderNeuroModulators {
        EncoderNeuroModulators {
            dopamine: (snap.workload_efficiency as f32).clamp(0.0, 1.0),
            cortisol: snap.thermal_stress().clamp(0.0, 1.0),
            acetylcholine: (snap.mem_util_pct / 100.0).clamp(0.0, 1.0),
            tempo: (0.5 + (snap.gpu_clock_mhz / 3000.0)).clamp(0.5, 2.0),
            fpga_stress: 0.0,
            market_volatility: ((snap.auxiliary_signal as f32).abs()).clamp(0.0, 1.0),
            mining_dopamine: ((snap.workload_efficiency as f32) - 0.5) * 1.6,
        }
    }

    fn encoded_output_to_stimuli(encoded: &spikenaut_encoder::types::EncodedOutput) -> [f32; SNN_INPUT_CHANNELS] {
        let mut stimuli = [0.0_f32; SNN_INPUT_CHANNELS];
        for spike in &encoded.spikes {
            let idx = spike.channel as usize;
            if idx < SNN_INPUT_CHANNELS {
                let delta = if spike.polarity { 1.0 } else { -1.0 };
                stimuli[idx] = (stimuli[idx] + delta).clamp(-1.0, 1.0);
            }
        }
        stimuli
    }

    /// Publish a [`TrainSignal`] to `SpikenautDistill.jl` via the ZMQ spine.
    /// No-op (always succeeds) when the `spine-zmq` feature is disabled.
    fn publish_train_signal(&self, signal: &TrainSignal) -> Result<()> {
        #[cfg(feature = "spine-zmq")]
        {
            use spikenaut_spine::SpinePublisher;
            let payload = serde_json::to_string(signal)?;
            let publisher = SpinePublisher::new(&self.config.spine_endpoint)
                .map_err(|e| HybridError::SpineInit(e.to_string()))?;
            publisher
                .publish(payload.as_bytes())
                .map_err(|e| HybridError::SpinePublish(e.to_string()))?;
        }
        let _ = signal;
        Ok(())
    }
}

// ── Neuromodulator bridge ─────────────────────────────────────────────────────

/// Translate `spikenaut_encoder::NeuroModulators` → `neuromod::NeuroModulators`.
///
/// - Shared fields (dopamine, cortisol, acetylcholine, tempo, mining_dopamine)
///   are mapped 1-to-1 with clamping.
/// - Encoder-exclusive fields (`fpga_stress`, `market_volatility`) are folded
///   into the composite cortisol signal.
fn bridge_modulators(enc: &EncoderNeuroModulators) -> neuromod::NeuroModulators {
    let composite_cortisol = (enc.cortisol + enc.market_volatility * 0.5).clamp(0.0, 1.0);
    neuromod::NeuroModulators {
        dopamine:        enc.dopamine.clamp(0.0, 1.0),
        cortisol:        composite_cortisol,
        acetylcholine:   enc.acetylcholine.clamp(0.0, 1.0),
        tempo:           enc.tempo.clamp(0.5, 2.0),
        mining_dopamine: enc.mining_dopamine.clamp(-0.8, 0.8),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OlmoeExecutionMode, ProjectionMode};
    use spikenaut_telemetry::TelemetrySnapshot;

    fn default_model() -> HybridModel {
        HybridModel::new(HybridConfig::default()).expect("default model init failed")
    }

    #[test]
    fn test_model_creation() {
        let model = default_model();
        assert_eq!(model.global_step(), 0);
        assert!(!model.olmoe_loaded());
    }

    #[test]
    fn test_forward_smoke() {
        let mut model = default_model();
        let out = model.forward(&TelemetrySnapshot::default()).unwrap();
        assert_eq!(out.firing_rates.len(), 16);
        assert_eq!(out.embedding.len(), EMBEDDING_DIM);
        assert!(out.expert_weights.is_some());
        assert!(out.selected_experts.is_some());
    }

    #[test]
    fn test_step_counter_increments() {
        let mut model = default_model();
        let snap = TelemetrySnapshot::default();
        model.forward(&snap).unwrap();
        model.forward(&snap).unwrap();
        assert_eq!(model.global_step(), 2);
    }

    #[test]
    fn test_reset_clears_step_counter() {
        let mut model = default_model();
        model.forward(&TelemetrySnapshot::default()).unwrap();
        model.reset();
        assert_eq!(model.global_step(), 0);
    }

    #[test]
    fn test_reset_clears_olmoe_spiking_state() {
        let cfg = HybridConfig {
            olmoe_execution_mode: OlmoeExecutionMode::SpikingSim,
            ..Default::default()
        };
        let mut model = HybridModel::new(cfg).unwrap();

        for _ in 0..8 {
            model.forward(&TelemetrySnapshot::default()).unwrap();
        }

        assert!(model.olmoe.has_state_activity());

        model.reset();

        assert!(!model.olmoe.has_state_activity());
    }

    #[test]
    fn test_invalid_snn_steps_zero() {
        let cfg = HybridConfig { snn_steps: 0, ..Default::default() };
        assert!(HybridModel::new(cfg).is_err());
    }

    #[test]
    fn test_invalid_top_k_exceeds_experts() {
        let cfg = HybridConfig { num_experts: 4, top_k_experts: 8, ..Default::default() };
        assert!(HybridModel::new(cfg).is_err());
    }

    #[test]
    fn test_train_step_non_negative_loss() {
        let mut model = default_model();
        let snap = TelemetrySnapshot {
            gpu_temp_c: 72.0, gpu_power_w: 280.0,
            cpu_tctl_c: 65.0, workload_throughput: 10.0, workload_efficiency: 0.7,
            ..Default::default()
        };
        let loss = model.train_step(&snap, &vec![0.1_f32; EMBEDDING_DIM]).unwrap();
        assert!(loss >= 0.0, "loss must be non-negative, got {loss}");
    }

    #[test]
    fn test_all_projection_modes() {
        for mode in [
            ProjectionMode::RateSum,
            ProjectionMode::TemporalHistogram,
            ProjectionMode::MembraneSnapshot,
            ProjectionMode::SpikingTernary,
        ] {
            let cfg = HybridConfig { projection_mode: mode, ..Default::default() };
            let mut model = HybridModel::new(cfg).unwrap();
            let out = model.forward(&TelemetrySnapshot::default()).unwrap();
            assert_eq!(out.embedding.len(), EMBEDDING_DIM, "mode: {mode:?}");
        }
    }

    #[test]
    fn test_bridge_neuromod_clamps() {
        let enc = EncoderNeuroModulators {
            dopamine: 1.5,           // clamped → 1.0
            cortisol: 0.4,
            acetylcholine: 0.7,
            tempo: 1.2,
            fpga_stress: 0.0,
            market_volatility: 0.6,
            mining_dopamine: -0.9,   // clamped → -0.8
        };
        let nm = bridge_modulators(&enc);
        assert!((nm.dopamine - 1.0).abs() < 1e-5);
        assert!(nm.cortisol <= 1.0);
        assert!((nm.mining_dopamine - (-0.8)).abs() < 1e-5);
    }
}
