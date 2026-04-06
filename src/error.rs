//! Error types for `spikenaut-hybrid`.
//!
//! All public functions in this crate return [`HybridError`] wrapped in a
//! [`Result`].  Downstream callers can match on variants to distinguish
//! configuration mistakes from I/O failures or runtime SNN errors.

use thiserror::Error;

/// Unified error type for the hybrid framework.
#[derive(Debug, Error)]
pub enum HybridError {
    // ── Configuration errors ──────────────────────────────────────────────
    /// A required field in [`HybridConfig`](crate::types::HybridConfig) was
    /// empty or out of range.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    // ── Model loading errors ──────────────────────────────────────────────
    /// The GGUF or safetensors file could not be opened / parsed.
    #[error("model load failed for '{path}': {reason}")]
    ModelLoad { path: String, reason: String },

    /// The model file format is not supported (e.g. wrong GGUF magic).
    #[error("unsupported model format: {0}")]
    UnsupportedFormat(String),

    /// A required tensor or layer was missing from the checkpoint.
    #[error("missing tensor '{name}' in model '{path}'")]
    MissingTensor { name: String, path: String },

    // ── Forward-pass errors ───────────────────────────────────────────────
    /// Input slice had the wrong length.
    #[error("input length mismatch: expected {expected}, got {got}")]
    InputLengthMismatch { expected: usize, got: usize },

    /// The SNN produced no spikes (silent network — likely a config problem).
    #[error("SNN produced no spikes after {steps} steps — network may be silent")]
    SilentNetwork { steps: usize },

    /// OLMoE forward pass returned an error.
    #[error("OLMoE forward pass failed: {0}")]
    OlmoeForward(String),

    // ── Spine / Julia IPC errors ──────────────────────────────────────────
    /// The ZMQ spine could not be initialised (e.g. libzmq not installed).
    #[error("spine initialisation failed: {0}")]
    SpineInit(String),

    /// Sending a loss signal to SpikenautDistill.jl failed mid-flight.
    #[error("spine publish failed: {0}")]
    SpinePublish(String),

    // ── I/O errors ────────────────────────────────────────────────────────
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    // ── JSON serialisation errors ─────────────────────────────────────────
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, HybridError>;
