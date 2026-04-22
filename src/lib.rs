//! # hybrid-fusion
//!
//! Pure-Rust master orchestrator for a hybrid transformer ↔ spiking
//! neural network stack. Zero Candle, zero CUDA, zero Julia.
//!
//! ## Pillars
//!
//! | Crate | Role |
//! |-------|------|
//! | [`cortex_tensor`] | Tensor + transformer + MoE math. |
//! | [`engram_parser`] | GGUF checkpoint parser (zero-dep). |
//! | [`neuromod`] | LIF / Izhikevich spiking dynamics (v0.4.0). |
//!
//! ## Data flow
//!
//! ```text
//! token_ids: &[u32]
//!      │
//!      ▼  cortex_tensor::TransformerLM::hidden_states
//! cortex_tensor::Tensor   [seq_len, dim]
//!      │
//!      ▼  projector::embed_to_stimuli_with_width   (pool → resize → tanh)
//! stimuli: Vec<f32>       strictly bounded in (-1, 1)
//!      │
//!      ▼  neuromod::SpikingNetwork::step(&stimuli, &modulators)
//! fired_neurons: Vec<usize>
//! ```
//!
//! ## Quick start
//!
//! ```no_run
//! use hybrid_fusion::{HybridConfig, HybridNetwork};
//!
//! let mut net = HybridNetwork::from_config(HybridConfig::tiny())?;
//! let out = net.forward(&[1u32, 2, 3, 4], None)?;
//! assert_eq!(out.stimuli.len(), net.snn.num_channels);
//! # Ok::<(), hybrid_fusion::HybridError>(())
//! ```

pub mod error;
pub mod hybrid;
pub mod projector;
pub mod types;

// ── Core re-exports ───────────────────────────────────────────────────────────

pub use error::{HybridError, Result};
pub use hybrid::HybridNetwork;
pub use types::{HybridConfig, HybridOutput};

// ── Convenience re-exports from the pure-Rust stack ───────────────────────────

/// Re-exported so downstream users don't need a direct `cortex-tensor`
/// dependency for the common tensor type.
pub use cortex_tensor::Tensor;

/// Re-exported `neuromod` neuromodulator struct — derived from telemetry /
/// market-data upstream of the orchestrator.
pub use neuromod::NeuroModulators;
