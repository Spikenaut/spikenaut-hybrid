//! # spikenaut-hybrid
//!
//! **Neuromorphic-ANN hybrid framework** — the top-level orchestrator for the
//! [SpikeLMo](https://github.com/Spikenaut/spikenaut-hybrid) fusion between
//! the Spikenaut SNN and [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct).
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │                       spikenaut-hybrid pipeline                           │
//! │                                                                           │
//! │  TelemetrySnapshot (spikenaut-telemetry)                                  │
//! │        │                                                                  │
//! │        ▼  spikenaut-encoder                                               │
//! │  NeuromodSensoryEncoder  →  [f32; 16] Poisson stimuli                     │
//! │        │                                                                  │
//! │        ▼  neuromod                                                        │
//! │  SpikingNetwork × snn_steps  →  spike_train + membrane_potentials         │
//! │        │                                                                  │
//! │        ▼  spikenaut-hybrid :: Projector                                   │
//! │  dense embedding  [EMBEDDING_DIM = 2048]                                  │
//! │        │                                                                  │
//! │        ▼  spikenaut-hybrid :: OLMoE  (frozen)                             │
//! │  OlmoeOutput { expert_weights, selected_experts, hidden }                 │
//! │        │                                                                  │
//! │        ▼  spikenaut-spine  (optional, spine-zmq feature)                  │
//! │  TrainSignal  ──►  SpikenautDistill.jl  (E-prop / OTTT on SNN only)      │
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick start
//!
//! ```no_run
//! use spikenaut_hybrid::{HybridConfig, HybridModel};
//! use spikenaut_telemetry::TelemetrySnapshot;
//!
//! // 1. Build config — empty model path means "stub mode" (no GPU/model needed)
//! let cfg = HybridConfig {
//!     olmoe_model_path: String::new(),   // stub; set to GGUF path for real inference
//!     snn_steps: 20,
//!     ..Default::default()
//! };
//!
//! // 2. Instantiate the hybrid
//! let mut model = HybridModel::new(cfg).unwrap();
//!
//! // 3. Feed telemetry
//! let snap = TelemetrySnapshot::default();
//! let output = model.forward(&snap).unwrap();
//!
//! println!("Firing rates:     {:?}", &output.firing_rates[..4]);
//! println!("Selected experts: {:?}", output.selected_experts);
//! ```
//!
//! ## Feature flags
//!
//! | Feature | What it adds |
//! |---------|-------------|
//! | `gguf` | GGUF Q5_K_M model parsing (pure Rust, no C++) |
//! | `safetensors` | BF16 safetensors loading |
//! | `spine-zmq` | ZMQ transport for Rust ↔ Julia training bridge |
//!
//! ## Crate layout
//!
//! | Module | Role |
//! |--------|------|
//! | [`types`] | `HybridConfig`, `HybridOutput`, `ProjectionMode`, `TrainSignal` |
//! | [`error`] | `HybridError`, `Result` |
//! | [`projector`] | Spike → dense embedding (the neuromorphic fusion bridge) |
//! | [`olmoe`] | Frozen OLMoE inference engine |
//! | [`hybrid`] | `HybridModel` top-level orchestrator |

// ── Modules ───────────────────────────────────────────────────────────────────

pub mod error;
pub mod hybrid;
pub mod olmoe;
pub mod projector;
pub mod types;

// ── Public re-exports ─────────────────────────────────────────────────────────

/// Main hybrid model — start here.
pub use hybrid::HybridModel;

/// Configuration and output types.
pub use types::{HybridConfig, HybridOutput, OlmoeExecutionMode, ProjectionMode, TrainSignal};
pub use types::{EMBEDDING_DIM, SNN_INPUT_CHANNELS};

/// Error type.
pub use error::{HybridError, Result};

/// Projector (re-exported for researchers who want to use it standalone).
pub use projector::Projector;

// ── Lower-level crate re-exports ──────────────────────────────────────────────
// Researchers can import these directly from spikenaut-hybrid without also
// adding the lower-level crates to their own Cargo.toml.

/// Re-export of `spikenaut-encoder` for convenience.
pub use spikenaut_encoder;

/// Re-export of `neuromod` for convenience.
pub use neuromod;

/// Re-export of `spikenaut-telemetry` for convenience.
pub use spikenaut_telemetry;

/// Re-export of `spikenaut-spine` for convenience.
pub use spikenaut_spine;
