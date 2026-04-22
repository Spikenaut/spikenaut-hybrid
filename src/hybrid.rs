//! Master orchestrator.
//!
//! `HybridNetwork` stitches together three pure-Rust dependencies:
//!
//! | Crate | Role |
//! |-------|------|
//! | [`cortex_tensor::transformer::TransformerLM`] | Transformer forward pass (no Candle). |
//! | [`crate::projector`] | Pool + bounded-squash LLM hidden state → SNN stimulus. |
//! | [`neuromod::SpikingNetwork`] | LIF + Izhikevich spiking dynamics. |
//!
//! GGUF weight loading is delegated to `engram_parser::load_gguf`; the
//! actual tensor-to-`TransformerLM` binding is left as a **TODO** until
//! `cortex-tensor` exposes a public loader. No stubbed / fabricated
//! weights are injected.
//!
//! ## Forward-pass data flow
//!
//! ```text
//!   token_ids: &[u32]
//!        │
//!        ▼  transformer.hidden_states(token_ids)
//!   cortex_tensor::Tensor  [seq_len, dim]
//!        │
//!        ▼  projector::embed_to_stimuli_with_width(&hidden, snn.num_channels)
//!   stimuli: Vec<f32>     (tanh-bounded, dynamic length)
//!        │
//!        ▼  snn.step(&stimuli, &modulators)
//!   fired_neurons: Vec<usize>
//! ```

use cortex_tensor::transformer::{TransformerConfig, TransformerLM};
use neuromod::{NeuroModulators, SpikingNetwork};

use crate::error::{HybridError, Result};
use crate::projector;
use crate::types::{HybridConfig, HybridOutput};

/// Master orchestrator over `cortex-tensor` + `neuromod`.
pub struct HybridNetwork {
    /// Pure-Rust transformer (Candle-free).
    pub transformer: TransformerLM,
    /// Spiking neural network (LIF + Izhikevich).
    pub snn: SpikingNetwork,
    /// Orchestrator config.
    config: HybridConfig,
    /// Monotonic forward-pass counter.
    global_step: u64,
}

impl HybridNetwork {
    /// Build a `HybridNetwork` from an explicit transformer + SNN.
    pub fn new(transformer: TransformerLM, snn: SpikingNetwork, config: HybridConfig) -> Self {
        Self {
            transformer,
            snn,
            config,
            global_step: 0,
        }
    }

    /// Build a `HybridNetwork` from a [`HybridConfig`] alone.
    ///
    /// Instantiates a fresh `TransformerLM` with randomised weights and a
    /// `SpikingNetwork` sized to the configured LIF / Izhikevich banks and
    /// input channels. Suitable for tests and stub runs.
    pub fn from_config(config: HybridConfig) -> Result<Self> {
        if config.snn_input_channels == 0 {
            return Err(HybridError::InvalidConfig(
                "snn_input_channels must be > 0".into(),
            ));
        }
        if config.snn_lif_neurons == 0 {
            return Err(HybridError::InvalidConfig(
                "snn_lif_neurons must be > 0".into(),
            ));
        }

        let transformer = TransformerLM::new(config.transformer.clone());
        let snn = SpikingNetwork::with_dimensions(
            config.snn_lif_neurons,
            config.snn_izh_neurons,
            config.snn_input_channels,
        );
        Ok(Self::new(transformer, snn, config))
    }

    /// Load transformer weights from a GGUF checkpoint.
    ///
    /// ⚠ **Incomplete:** this function parses the GGUF layout via
    /// `engram_parser::load_gguf` and reports its metadata through
    /// [`HybridError::ModelLoad`] / [`HybridError::UnsupportedFormat`],
    /// **but does not yet bind tensors into `TransformerLM`**. Weight
    /// binding will be added in a follow-up PR once `cortex-tensor`
    /// exposes a public GGUF loader. No fabricated / zero-weights are
    /// substituted — callers receive an explicit
    /// [`HybridError::UnsupportedFormat`] until the loader is wired.
    pub fn load_weights_from_gguf(&mut self, path: &str) -> Result<()> {
        let layout = engram_parser::load_gguf(path).map_err(|e| HybridError::GgufParse(format!("{e:?}")))?;
        let arch = layout.metadata.architecture().to_string();

        // TODO(cortex-tensor): wire the tensor directory into TransformerLM
        //   once `cortex_tensor::transformer::TransformerLM` gains a public
        //   loader (e.g. `TransformerLM::from_gguf_layout(&layout)`).
        //   Until then, surface a clean error so callers don't receive
        //   silently-broken random weights.
        Err(HybridError::UnsupportedFormat(format!(
            "GGUF weight binding not yet implemented (path='{path}', arch='{arch}'). \
             Parsed {} tensors; waiting on cortex-tensor public loader.",
            layout.tensors.len()
        )))
    }

    /// Run the orchestrator for a single prompt.
    ///
    /// 1. Transformer → hidden state `[seq_len, dim]`.
    /// 2. Projector → `Vec<f32>` stimulus of length `self.snn.num_channels`
    ///    (pool → resize → **tanh**).
    /// 3. SNN → fired neuron indices.
    ///
    /// `modulators` defaults to `NeuroModulators::default()` when `None`.
    pub fn forward(
        &mut self,
        token_ids: &[u32],
        modulators: Option<NeuroModulators>,
    ) -> Result<HybridOutput> {
        if token_ids.is_empty() {
            return Err(HybridError::InputLengthMismatch {
                expected: 1,
                got: 0,
            });
        }
        if token_ids.len() > self.transformer.config.max_seq_len {
            return Err(HybridError::InputLengthMismatch {
                expected: self.transformer.config.max_seq_len,
                got: token_ids.len(),
            });
        }

        // ── 1. Transformer forward (hidden state) ─────────────────────────
        let hidden = self.transformer.hidden_states(token_ids);

        // Mean-pool across the sequence axis → `dim`-long embedding. This
        // is also what the projector does internally, but we keep a copy
        // for the `HybridOutput.embedding` field so downstream callers can
        // inspect the raw pooled hidden state without re-running the net.
        let embedding = pool_embedding(&hidden, self.transformer.config.dim);

        // ── 2. Projector → SNN stimulus (dynamic width, bounded) ──────────
        let snn_width = self.snn.num_channels;
        let stimuli = projector::embed_to_stimuli_with_width(&hidden, snn_width);

        // ── 3. SNN step ───────────────────────────────────────────────────
        let modulators = modulators.unwrap_or_default();
        let fired_neurons = self.snn.step(&stimuli, &modulators)?;

        self.global_step = self.global_step.saturating_add(1);

        Ok(HybridOutput {
            embedding,
            stimuli,
            fired_neurons,
            global_step: self.global_step,
        })
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn config(&self) -> &HybridConfig {
        &self.config
    }
    pub fn global_step(&self) -> u64 {
        self.global_step
    }
    pub fn transformer_config(&self) -> &TransformerConfig {
        &self.transformer.config
    }
    pub fn reset(&mut self) {
        self.global_step = 0;
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Mean-pool a `[seq, dim]` or 1-D tensor down to a `Vec<f32>` of length `dim`.
/// Kept private so `HybridOutput.embedding` always matches the transformer's
/// declared hidden size, regardless of the SNN's input width.
fn pool_embedding(hidden: &cortex_tensor::Tensor, dim: usize) -> Vec<f32> {
    if hidden.ndim() == 1 {
        return hidden.data().to_vec();
    }
    if hidden.ndim() == 2 {
        let shape = hidden.shape();
        let seq = shape[0].max(1);
        let hdim = shape[1];
        let take = dim.min(hdim);
        let data = hidden.data();
        let mut pooled = vec![0.0f32; take];
        for t in 0..seq {
            let row = &data[t * hdim..(t + 1) * hdim];
            for (i, v) in row.iter().take(take).enumerate() {
                pooled[i] += *v;
            }
        }
        let inv = 1.0 / seq as f32;
        for v in &mut pooled {
            *v *= inv;
        }
        return pooled;
    }
    hidden.data().iter().copied().take(dim).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HybridConfig;

    fn build_network() -> HybridNetwork {
        HybridNetwork::from_config(HybridConfig::tiny()).expect("tiny config must build")
    }

    #[test]
    fn test_forward_shape_and_bounds() {
        let mut net = build_network();
        let token_ids = vec![1u32, 2, 3, 4];
        let out = net
            .forward(&token_ids, None)
            .expect("forward must succeed on tiny config");

        // embedding == transformer dim
        assert_eq!(out.embedding.len(), net.transformer.config.dim);
        // stimuli == snn num_channels (decoupled from transformer dim)
        assert_eq!(out.stimuli.len(), net.snn.num_channels);
        // tanh-bounded
        for v in &out.stimuli {
            assert!(v.abs() <= 1.0, "stimulus out of (-1, 1): {v}");
        }
        assert_eq!(out.global_step, 1);
    }

    #[test]
    fn test_forward_rejects_empty_prompt() {
        let mut net = build_network();
        let err = net.forward(&[], None).unwrap_err();
        matches!(err, HybridError::InputLengthMismatch { .. });
    }

    #[test]
    fn test_forward_rejects_over_long_prompt() {
        let mut net = build_network();
        let too_long = vec![0u32; net.transformer.config.max_seq_len + 1];
        let err = net.forward(&too_long, None).unwrap_err();
        matches!(err, HybridError::InputLengthMismatch { .. });
    }

    #[test]
    fn test_global_step_increments() {
        let mut net = build_network();
        let ids = vec![7u32, 8];
        net.forward(&ids, None).unwrap();
        net.forward(&ids, None).unwrap();
        assert_eq!(net.global_step(), 2);
        net.reset();
        assert_eq!(net.global_step(), 0);
    }

    #[test]
    fn test_invalid_config_rejected() {
        let mut cfg = HybridConfig::tiny();
        cfg.snn_input_channels = 0;
        assert!(HybridNetwork::from_config(cfg).is_err());
    }

    #[test]
    fn test_load_weights_from_gguf_surfaces_error_when_path_bogus() {
        let mut net = build_network();
        let err = net
            .load_weights_from_gguf("/nonexistent/path/model.gguf")
            .unwrap_err();
        matches!(err, HybridError::GgufParse(_));
    }

    #[test]
    fn test_snn_width_independent_from_transformer_dim() {
        // Force a mismatch: transformer.dim == 128, snn input == 7.
        let mut cfg = HybridConfig::tiny();
        cfg.snn_input_channels = 7;
        let mut net = HybridNetwork::from_config(cfg).unwrap();
        let out = net.forward(&[0u32, 1, 2], None).unwrap();
        assert_eq!(out.stimuli.len(), 7);
        assert_eq!(out.embedding.len(), net.transformer.config.dim);
    }
}
