//! # hybrid_telemetry
//!
//! End-to-end demo of the SpikeLMo neuromorphic-ANN hybrid pipeline.
//!
//! Simulates 100 seconds of mining/HFT telemetry, feeds each snapshot through
//! the full pipeline (SNN → Projector → OLMoE), prints per-step diagnostics,
//! and demonstrates the training-step hook that sends E-prop signals to Julia.
//!
//! Run with:
//! ```bash
//! cargo run --example hybrid_telemetry
//! ```
//!
//! To use a real OLMoE checkpoint (GGUF, ~4 GB):
//! ```bash
//! OLMOE_PATH=/models/OLMoE-1B-7B-Q5_K_M.gguf cargo run --example hybrid_telemetry --features gguf
//! ```

use spikenaut_hybrid::{
    EMBEDDING_DIM, HybridConfig, HybridModel, OlmoeExecutionMode, ProjectionMode,
};
use spikenaut_telemetry::TelemetrySnapshot;

fn main() -> spikenaut_hybrid::Result<()> {
    // ── 1. Configuration ─────────────────────────────────────────────────────
    let model_path = std::env::var("OLMOE_PATH").unwrap_or_default();
    let olmoe_execution_mode = match std::env::var("OLMOE_MODE")
        .unwrap_or_else(|_| "stub".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "dense" => OlmoeExecutionMode::DenseSim,
        "spiking" => OlmoeExecutionMode::SpikingSim,
        _ => OlmoeExecutionMode::StubUniform,
    };
    if model_path.is_empty() {
        println!("ℹ  OLMOE_PATH not set — running in stub mode (no LLM checkpoint needed).");
        println!("   Set OLMOE_PATH=/path/to/OLMoE-1B-7B-Q5_K_M.gguf for real inference.\n");
    } else {
        println!("🧠 Loading OLMoE from: {model_path}\n");
    }

    let cfg = HybridConfig {
        olmoe_model_path: model_path,
        snn_steps: 20,
        context_length: 512,
        num_experts: 8,
        top_k_experts: 1,
        olmoe_execution_mode,
        initial_dopamine: 0.6,
        projection_mode: ProjectionMode::RateSum,
        ..Default::default()
    };

    println!("⚙  Config:");
    println!("   SNN steps / forward: {}", cfg.snn_steps);
    println!("   Projection mode    : {:?}", cfg.projection_mode);
    println!("   OLMoE mode         : {:?}", cfg.olmoe_execution_mode);
    println!("   OLMoE experts       : {} (top-{})", cfg.num_experts, cfg.top_k_experts);
    println!();

    // ── 2. Build the hybrid model ─────────────────────────────────────────────
    let mut model = HybridModel::new(cfg)?;
    println!(
        "✅ HybridModel ready  |  OLMoE loaded: {}  |  step={}",
        model.olmoe_loaded(),
        model.global_step()
    );
    println!();

    // ── 3. Simulate telemetry stream ──────────────────────────────────────────
    let n_steps = 100usize;
    println!("🔄 Running {n_steps} telemetry steps ...\n");
    println!(
        "{:>5}  {:>8}  {:>10}  {:>10}  {:>12}  {:>8}",
        "step", "spikes", "top_expert", "embed_rms", "loss", "temp°C"
    );
    println!("{}", "─".repeat(64));

    let mut total_loss = 0.0_f32;

    for t in 0..n_steps {
        // Synthesise realistic telemetry for a mining rig under variable load
        let phase = t as f32 / n_steps as f32;
        let (gpu_temp, gpu_power, dynex_hash) = simulate_mining_telemetry(phase);

        let snap = TelemetrySnapshot {
            timestamp_ms: (t as u64) * 1000,
            gpu_temp_c: gpu_temp,
            gpu_power_w: gpu_power,
            gpu_clock_mhz: 1800.0 + phase * 600.0,
            mem_util_pct: (55.0 + phase * 35.0).clamp(0.0, 100.0),
            cpu_tctl_c: 55.0 + (phase * std::f32::consts::PI).sin() * 15.0,
            cpu_package_power_w: 80.0 + phase * 40.0,
            workload_throughput: dynex_hash,
            workload_efficiency: (0.5 + 0.3 * (phase * std::f32::consts::TAU).cos()) as f64,
            auxiliary_signal: (phase * 3.0).sin().abs() as f64,
            ..Default::default()
        };

        // Forward pass
        let output = model.forward(&snap)?;

        // Embedding RMS
        let embed_rms = {
            let sum_sq: f32 = output.embedding.iter().map(|v| v * v).sum();
            (sum_sq / EMBEDDING_DIM as f32).sqrt()
        };

        // Spike count across all steps
        let total_spikes: usize = output.spike_train.iter().map(|s| s.len()).sum();

        // Top expert
        let top_expert = output.selected_experts
            .as_ref()
            .and_then(|e| e.first().copied())
            .unwrap_or(0);

        // Training step with synthetic target (mean-field of embedding)
        let mean_embed = output.embedding.iter().sum::<f32>() / EMBEDDING_DIM as f32;
        let target = vec![mean_embed * 0.9; EMBEDDING_DIM]; // simple regression target
        let loss = model.train_step(&snap, &target)?;
        total_loss += loss;

        // Print diagnostics every 10 steps
        if t % 10 == 0 {
            println!(
                "{:>5}  {:>8}  {:>10}  {:>10.4}  {:>12.6}  {:>7.1}°C",
                t + 1,
                total_spikes,
                top_expert,
                embed_rms,
                loss,
                gpu_temp
            );
        }
    }

    // ── 4. Summary ────────────────────────────────────────────────────────────
    println!("{}", "─".repeat(64));
    println!();
    println!("📊 Summary after {n_steps} steps:");
    println!(
        "   Avg loss         : {:.6}",
        total_loss / n_steps as f32
    );
    println!("   Global step      : {}", model.global_step());
    println!("   SNN neuron count : {}", model.snn().neurons.len());
    println!(
        "   Proj dims        : {:?}",
        model.projector_mut().dims()
    );
    println!();
    println!("✨ Done!  Pipe spike_train + embedding → SpikenautDistill.jl for E-prop training.");
    Ok(())
}

// ── Synthetic telemetry generator ─────────────────────────────────────────────

/// Simulate a Dynex mining rig moving through warm-up → full load → throttle.
///
/// Returns `(gpu_temp_c, gpu_power_w, dynex_hashrate_mh)`.
fn simulate_mining_telemetry(phase: f32) -> (f32, f32, f64) {
    use std::f32::consts::PI;

    // GPU temperature: ramp up then plateau with noise
    let base_temp = 55.0 + phase * 30.0;
    let noise = (phase * 17.3 * PI).sin() * 2.0;
    let gpu_temp = (base_temp + noise).clamp(40.0, 92.0);

    // GPU power follows temperature with a lag
    let gpu_power = (200.0 + phase * 200.0 + (phase * PI * 2.0).sin() * 20.0)
        .clamp(100.0_f32, 450.0);

    // Dynex hashrate: peaks at mid-phase (optimal temp range)
    let optimal = 1.0 - (phase - 0.5).abs() * 2.0;
    let dynex_hash = (0.005 + optimal * 0.010) as f64;

    (gpu_temp, gpu_power, dynex_hash)
}
