//! Error types for `hybrid-fusion`.
//!
//! All public functions in this crate return [`HybridError`] wrapped in a
//! [`Result`]. The variants are intentionally minimal after the Candle /
//! `spike-lmo` purge — only errors that can actually originate from the
//! pure-Rust stack (`cortex-tensor`, `engram-parser`, `neuromod`) remain.

use thiserror::Error;

/// Unified error type for the hybrid orchestrator.
#[derive(Debug, Error)]
pub enum HybridError {
    // ── Configuration errors ──────────────────────────────────────────────
    /// A required field in [`HybridConfig`](crate::types::HybridConfig) was
    /// empty, zero, or otherwise out of range.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    // ── Model loading errors ──────────────────────────────────────────────
    /// The GGUF checkpoint file could not be opened or parsed.
    #[error("model load failed for '{path}': {reason}")]
    ModelLoad { path: String, reason: String },

    /// The checkpoint format is not supported (e.g. wrong GGUF magic).
    #[error("unsupported model format: {0}")]
    UnsupportedFormat(String),

    /// A required tensor or layer was missing from the checkpoint.
    #[error("missing tensor '{name}' in model '{path}'")]
    MissingTensor { name: String, path: String },

    /// GGUF header/metadata parse failure surfaced from `engram-parser`.
    #[error("GGUF parse failed: {0}")]
    GgufParse(String),

    // ── Forward-pass errors ───────────────────────────────────────────────
    /// Input slice had the wrong length.
    #[error("input length mismatch: expected {expected}, got {got}")]
    InputLengthMismatch { expected: usize, got: usize },

    /// `neuromod::SpikingNetwork::step` returned an error.
    #[error("SNN step failed: {0}")]
    SnnStep(String),

    // ── I/O errors ────────────────────────────────────────────────────────
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    // ── JSON serialisation errors ─────────────────────────────────────────
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, HybridError>;

// ── Cross-crate error bridges ─────────────────────────────────────────────

impl From<neuromod::StepError> for HybridError {
    fn from(err: neuromod::StepError) -> Self {
        // `StepError` has no `Display` impl — use `Debug` formatting.
        HybridError::SnnStep(format!("{err:?}"))
    }
}
