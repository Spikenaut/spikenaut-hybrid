# hybrid-fusion

**Pure-Rust master orchestrator for a hybrid transformer ↔ spiking neural
network stack.** Zero Candle, zero CUDA, zero Julia.

`hybrid-fusion` wires together three focused crates:

| Crate | Role |
|-------|------|
| [`cortex-tensor`](https://github.com/Limen-Neural/cortex-tensor) | Tensor, transformer, and MoE math. |
| [`engram-parser`](https://github.com/Limen-Neural/engram-parser) | Zero-dep GGUF checkpoint parser. |
| [`neuromod`](https://github.com/Limen-Neural/neuromod) 0.4.0 | LIF / Izhikevich spiking dynamics. |

The crate exposes a single orchestrator, `HybridNetwork`, that pipes a
prompt through the transformer, reduces the resulting hidden state into a
bounded stimulus vector, and steps a `neuromod::SpikingNetwork`.

## Forward-pass data flow

```text
token_ids: &[u32]
     │
     ▼  cortex_tensor::TransformerLM::hidden_states
cortex_tensor::Tensor   [seq_len, dim]
     │
     ▼  projector::embed_to_stimuli_with_width   (pool → resize → tanh)
stimuli: Vec<f32>       ∈ [-1, 1], length == snn.num_channels
     │
     ▼  neuromod::SpikingNetwork::step(&stimuli, &modulators)
fired_neurons: Vec<usize>
```

The `tanh` squash is applied **after** pooling and resizing so the values
fed into the SNN are always bounded — `neuromod` requires bounded input
to prevent membrane-voltage blow-ups.

## Quick start

```rust
use hybrid_fusion::{HybridConfig, HybridNetwork, NeuroModulators};

let mut net = HybridNetwork::from_config(HybridConfig::tiny())?;

let token_ids = [1u32, 2, 3, 4];
let modulators = NeuroModulators::default();

let out = net.forward(&token_ids, Some(modulators))?;

assert_eq!(out.embedding.len(), net.transformer.config.dim);
assert_eq!(out.stimuli.len(), net.snn.num_channels);
# Ok::<(), hybrid_fusion::HybridError>(())
```

Run the bundled telemetry demo:

```bash
cargo run --example hybrid_telemetry
```

## Public surface

| Item | Purpose |
|------|---------|
| `HybridNetwork::new(transformer, snn, config)` | Explicit constructor. |
| `HybridNetwork::from_config(HybridConfig)` | Build from a config only (random weights). |
| `HybridNetwork::load_weights_from_gguf(path)` | Parses GGUF layout via `engram-parser`. **TODO:** tensor binding pending a public `cortex-tensor` loader — surfaces an explicit `HybridError::UnsupportedFormat` rather than fabricating weights. |
| `HybridNetwork::forward(&[u32], Option<NeuroModulators>)` | Transformer → projector → SNN. |
| `projector::embed_to_stimuli(&Tensor)` | Pool + tanh-squash, width = embedding dim. |
| `projector::embed_to_stimuli_with_width(&Tensor, width)` | Pool → resize → tanh to a caller-chosen width. |
| `HybridConfig::tiny()` / `::olmo_1b()` | Predefined transformer + SNN shapes. |

All dimensions are **dynamic**. No hardcoded `16`-channel bottlenecks,
no fixed `EMBEDDING_DIM = 2048`, no `NUM_INPUT_CHANNELS` constants leak
through the public API.

## What this crate is **not**

- **Not** a training framework. `neuromod` handles spike dynamics; any
  learning loop (e.g. reward-modulated STDP) lives upstream.
- **Not** a tokenizer. Callers pass `&[u32]` token IDs directly.
- **Not** a Candle bridge. The previous `spike-lmo` / Candle-era fusion
  engine, the `SnnLlmFusion` math, the `OLMoE` MoE wrapper, the
  `DenseModel` trait, and the `spikenaut-spine` ZMQ distill publisher
  have all been removed from this crate.

## Status

Experimental. API is expected to change as `cortex-tensor` exposes its
GGUF loader and as the projector grows richer pooling modes.

## License

GPL-3.0-or-later.
