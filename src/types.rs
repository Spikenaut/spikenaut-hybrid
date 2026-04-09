//! Public data types for `spikenaut-hybrid`.
//!
//! Two key structs dominate:
//!
//! * [`HybridConfig`] — everything needed to build a [`HybridModel`](crate::hybrid::HybridModel).
//! * [`HybridOutput`] — the combined result of one forward pass through the
//!   entire Telemetry → SNN → Projector → OLMoE pipeline.

use serde::{Deserialize, Serialize};

// ── Channel counts ─────────────────────────────────────────────────────────────

/// Number of input channels produced by `spikenaut-encoder` and consumed by
/// the Spikenaut SNN (`neuromod::SpikingNetwork`).
pub const SNN_INPUT_CHANNELS: usize = 16;

/// Dimensionality of the dense embedding the [`Projector`](crate::projector::Projector)
/// hands to OLMoE.  Must match the model's hidden size (2048 for OLMoE-1B-7B).
pub const EMBEDDING_DIM: usize = 2048;

// ── HybridConfig ──────────────────────────────────────────────────────────────

/// Top-level configuration for the neuromorphic-ANN hybrid.
///
/// # Example
/// ```no_run
/// use spikenaut_hybrid::HybridConfig;
///
/// let cfg = HybridConfig {
///     olmoe_model_path: "/models/OLMoE-1B-7B-Q5_K_M.gguf".into(),
///     context_length: 512,
///     snn_steps: 20,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    // ── OLMoE settings ───────────────────────────────────────────────────────
    /// Path to the Q5_K_M GGUF file (or safetensors directory when the
    /// `safetensors` feature is enabled).
    ///
    /// Set to an empty string to run in **stub mode** (projector output is
    /// returned without any LLM forward pass — useful for SNN-only research).
    pub olmoe_model_path: String,

    /// Maximum context length passed to OLMoE.  Defaults to 512.
    pub context_length: usize,

    /// Number of OLMoE experts per MoE layer (8 for OLMoE-1B-7B).
    pub num_experts: usize,

    /// Number of experts activated per token (top-k routing, default 1).
    pub top_k_experts: usize,

    /// Execution mode used by OLMoE.
    pub olmoe_execution_mode: OlmoeExecutionMode,

    // ── SNN settings ─────────────────────────────────────────────────────────
    /// Number of SNN time-steps to simulate per forward call.
    /// Higher values give richer spike statistics at the cost of latency.
    pub snn_steps: usize,

    /// Dopamine baseline injected into the SNN at startup.
    /// Overridden live by telemetry once the model is running.
    pub initial_dopamine: f32,

    // ── Projector settings ────────────────────────────────────────────────────
    /// Projection strategy used to turn spike activity into a dense embedding.
    pub projection_mode: ProjectionMode,

    // ── Spine (Rust ↔ Julia) settings ─────────────────────────────────────────
    /// ZMQ IPC endpoint for the SpikenautDistill.jl training bridge.
    /// Only used when the `spine-zmq` feature is enabled.
    pub spine_endpoint: String,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            olmoe_model_path: String::new(), // stub mode by default
            context_length: 512,
            num_experts: 8,
            top_k_experts: 1,
            olmoe_execution_mode: OlmoeExecutionMode::StubUniform,
            snn_steps: 20,
            initial_dopamine: 0.5,
            projection_mode: ProjectionMode::RateSum,
            spine_endpoint: "ipc:///tmp/spikenaut_hybrid.ipc".into(),
        }
    }
}

// ── ProjectionMode ─────────────────────────────────────────────────────────────

/// Strategy used by the [`Projector`](crate::projector::Projector) to convert
/// spike activity into a dense embedding for OLMoE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProjectionMode {
    /// **Rate-sum**: sum spike counts per neuron, then project via a learned
    /// linear layer.  Fast and robust for rate-coded inputs.
    #[default]
    RateSum,

    /// **Temporal histogram**: bin spikes across time-steps and flatten.
    /// Preserves more temporal structure for time-sensitive streams.
    TemporalHistogram,

    /// **Membrane potential**: use the post-step membrane potentials directly
    /// instead of binary spikes — a smooth, differentiable approximation.
    MembraneSnapshot,

    /// **Spiking ternary**: GIF membrane integration + ternary event output
    /// (-1.0 / 0.0 / 1.0).  Sparse, event-driven embedding; the first real
    /// SNN-logic quantized bridge to OLMoE (Optimal Brain Spiking style).
    /// Membrane state persists across calls — reset via
    /// [`Projector::reset_membrane`](crate::projector::Projector::reset_membrane).
    SpikingTernary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OlmoeExecutionMode {
    #[default]
    StubUniform,
    DenseSim,
    SpikingSim,
}

// ── HybridOutput ──────────────────────────────────────────────────────────────

/// The combined output of one [`HybridModel::forward`](crate::hybrid::HybridModel::forward)
/// call through the full Telemetry → SNN → Projector → OLMoE pipeline.
///
/// Fields are populated left-to-right through the pipeline; later stages are
/// `Option` so the struct is still useful when running in stub/SNN-only mode.
#[derive(Debug, Clone)]
pub struct HybridOutput {
    // ── SNN layer ────────────────────────────────────────────────────────────
    /// Spike train from the Spikenaut SNN: one `Vec<usize>` per time-step,
    /// containing the indices of neurons that fired in that step.
    pub spike_train: Vec<Vec<usize>>,

    /// Firing rates per neuron averaged over all SNN time-steps `[0, 1]`.
    pub firing_rates: Vec<f32>,

    /// Membrane potentials after the final SNN time-step.
    pub membrane_potentials: Vec<f32>,

    // ── Projector layer ───────────────────────────────────────────────────────
    /// Dense embedding produced by the [`Projector`](crate::projector::Projector)
    /// from the spike activity.  Shape: `[EMBEDDING_DIM]`.
    pub embedding: Vec<f32>,

    // ── OLMoE layer ───────────────────────────────────────────────────────────
    /// Expert routing probabilities from the MoE gating network.
    /// Shape: `[num_experts]`.  `None` in stub mode.
    pub expert_weights: Option<Vec<f32>>,

    /// Index of the top-k experts selected by the router.  `None` in stub mode.
    pub selected_experts: Option<Vec<usize>>,

    /// Decoded text output from OLMoE (if `generate = true` was requested).
    pub reasoning: Option<String>,
}

// ── TrainSignal ───────────────────────────────────────────────────────────────

/// Loss signal sent to `SpikenautDistill.jl` via the spine after each
/// [`HybridModel::train_step`](crate::hybrid::HybridModel::train_step) call.
///
/// Julia unpacks this to update only the SNN side via E-prop / OTTT;
/// the OLMoE weights remain frozen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainSignal {
    /// Wall-clock timestamp (ms since UNIX epoch).
    pub timestamp_ms: u64,

    /// Scalar prediction loss from OLMoE output vs. target.
    pub prediction_loss: f32,

    /// Thermal/power penalty term (cortisol proxy).
    pub thermal_penalty: f32,

    /// Per-neuron eligibility traces accumulated during the forward pass.
    /// Sent as flat `f32` array; Julia reshapes to `(16,)`.
    pub eligibility_traces: Vec<f32>,

    /// Current SNN step counter (global clock for Julia's LSM).
    pub global_step: i64,
}
