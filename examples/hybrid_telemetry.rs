//! # hybrid_telemetry
//!
//! Demonstrates the pure-Rust `hybrid-fusion` orchestrator:
//! `cortex-tensor` transformer → projector → `neuromod` SNN.
//!
//! Run:
//! ```bash
//! cargo run --example hybrid_telemetry
//! ```

use hybrid_fusion::{HybridConfig, HybridNetwork, NeuroModulators};

fn main() -> hybrid_fusion::Result<()> {
    // ── 1. Build the orchestrator from a tiny CI-scale config ────────────────
    let cfg = HybridConfig::tiny();
    println!(
        "✅ config | transformer.dim = {} | snn_channels = {} | lif = {} | izh = {}",
        cfg.transformer.dim, cfg.snn_input_channels, cfg.snn_lif_neurons, cfg.snn_izh_neurons,
    );

    let mut net = HybridNetwork::from_config(cfg)?;
    println!(
        "✅ HybridNetwork ready | transformer params ≈ {} | snn channels = {}\n",
        net.transformer.param_count(),
        net.snn.num_channels,
    );

    // ── 2. Fake telemetry → NeuroModulators → forward loop ──────────────────
    let n_steps = 10usize;
    let token_ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    println!("🔄 Running {n_steps} steps on {} token prompt ...\n", token_ids.len());
    println!(
        "{:>5}  {:>10}  {:>12}  {:>9}",
        "step", "stim[0]", "fired_count", "temp°C"
    );
    println!("{}", "─".repeat(48));

    for t in 0..n_steps {
        let gpu_temp = 60.0 + t as f32 * 1.5;

        // Derive neuromodulators from raw telemetry readings.
        let modulators = NeuroModulators {
            dopamine: (gpu_temp / 100.0).clamp(0.0, 1.0),
            cortisol: ((gpu_temp - 60.0) / 40.0).clamp(0.0, 1.0),
            acetylcholine: 0.7,
            tempo: 1.0,
            aux_dopamine: 0.5,
        };

        let out = net.forward(&token_ids, Some(modulators))?;

        println!(
            "{:>5}  {:>10.4}  {:>12}  {:>6.1}°C",
            out.global_step,
            out.stimuli[0],
            out.fired_neurons.len(),
            gpu_temp,
        );
    }

    println!("{}", "─".repeat(48));
    println!(
        "\n📊 final global_step = {} | embedding width = {}",
        net.global_step(),
        net.transformer.config.dim,
    );
    println!("✨ Done! Pure-Rust transformer → projector → SNN loop complete.");
    Ok(())
}
