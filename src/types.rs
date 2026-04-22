//! Public data types for `hybrid-fusion`.
//!
//! Two structs only, after the Candle / `spike-lmo` purge:
//!
//! * [`HybridConfig`] — orchestrator config: transformer dims + SNN dims.
//! * [`HybridOutput`] — the result of one `HybridNetwork::forward` call.
//!
//! Every legacy type that referenced `candle_core::Tensor`, `SnnLlmFusion`,
//! `DenseModel`, MoE expert routing, distill signals, or ZMQ spine payloads
//! has been removed.

use cortex_tensor::transformer::TransformerConfig;
use serde::{Deserialize, Serialize};

// ── HybridConfig ──────────────────────────────────────────────────────────────

/// Orchestrator configuration.
///
/// The transformer side is delegated entirely to `cortex_tensor::TransformerConfig`,
/// so shape/depth choices live there. The SNN side is described by three
/// dimensions consumed by `neuromod::SpikingNetwork::with_dimensions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Transformer shape (vocab, dim, heads, layers, ff_dim, max_seq_len).
    pub transformer: TransformerConfig,

    /// Number of LIF neurons in the spiking bank.
    pub snn_lif_neurons: usize,

    /// Number of Izhikevich neurons in the adaptive bank.
    pub snn_izh_neurons: usize,

    /// Number of input channels the SNN expects per `step`. The projector
    /// will dynamically resize embeddings to match this width.
    pub snn_input_channels: usize,
}

impl HybridConfig {
    /// A tiny preset suitable for CI / tests.
    ///
    /// Uses `TransformerConfig::tiny()` on the LLM side and a modestly-sized
    /// spiking bank on the SNN side. No hardcoded 2048 / 16 bottlenecks —
    /// callers are free to override any field.
    pub fn tiny() -> Self {
        Self {
            transformer: TransformerConfig::tiny(),
            snn_lif_neurons: 32,
            snn_izh_neurons: 8,
            snn_input_channels: 64,
        }
    }

    /// An OLMo-1B scale preset on the LLM side.
    pub fn olmo_1b() -> Self {
        Self {
            transformer: TransformerConfig::olmo_1b(),
            snn_lif_neurons: 128,
            snn_izh_neurons: 32,
            snn_input_channels: 256,
        }
    }
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self::tiny()
    }
}

// ── HybridOutput ──────────────────────────────────────────────────────────────

/// Result of one [`HybridNetwork::forward`](crate::HybridNetwork::forward) call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridOutput {
    /// Mean-pooled (over the sequence axis) hidden state emitted by the
    /// transformer, flattened to `Vec<f32>` of length `transformer.dim`.
    pub embedding: Vec<f32>,

    /// Tanh-bounded stimulus vector fed into the SNN. Length equals
    /// `config.snn_input_channels`.
    pub stimuli: Vec<f32>,

    /// Indices of neurons that fired during this step.
    pub fired_neurons: Vec<usize>,

    /// Monotonic step counter maintained by the orchestrator.
    pub global_step: u64,
}
