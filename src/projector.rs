//! Pure-Rust embedding → SNN-stimulus adapter.
//!
//! Replaces the legacy Xavier-weighted, SNN-to-embedding `Projector` with a
//! minimal **read-only mathematical adapter**. No learnable state, no
//! Candle / `candle_core::Tensor` imports, no `spike-lmo` coupling.
//!
//! # Pipeline
//!
//! ```text
//! cortex_tensor::Tensor  ──▶  mean-pool over seq   (if 2-D [seq, dim])
//!                         ──▶  resample to `snn_width`  (stride-pool or pad)
//!                         ──▶  tanh()  ← strictly bounded in (-1, 1)
//!                         ──▶  Vec<f32>  (length == snn_width)
//! ```
//!
//! The `tanh` squash is applied **after** the pool + resize step so the
//! output range is guaranteed to live in `(-1, 1)` regardless of the
//! transformer's raw activation magnitudes. `neuromod::SpikingNetwork`
//! requires bounded input to prevent membrane-voltage blow-ups.

use cortex_tensor::Tensor;

/// Mean-pool a `[seq, dim]` tensor into a `[dim]`-long `Vec<f32>`.
///
/// 1-D tensors are returned as-is. Higher-rank tensors are flattened to a
/// single row. No tanh is applied here — this is a raw pooling helper.
fn mean_pool(embedding: &Tensor) -> Vec<f32> {
    match embedding.ndim() {
        0 | 1 => embedding.data().to_vec(),
        2 => {
            let shape = embedding.shape();
            let seq = shape[0].max(1);
            let dim = shape[1];
            let data = embedding.data();
            let mut pooled = vec![0.0f32; dim];
            for t in 0..seq {
                let row = &data[t * dim..(t + 1) * dim];
                for (i, v) in row.iter().enumerate() {
                    pooled[i] += *v;
                }
            }
            let inv = 1.0 / seq as f32;
            for v in &mut pooled {
                *v *= inv;
            }
            pooled
        }
        _ => {
            // Fallback: treat the whole buffer as a flat vector.
            embedding.data().to_vec()
        }
    }
}

/// Resize a 1-D vector to exactly `target_width` entries by stride-pooling
/// (when `src.len() > target_width`) or zero-padding (when shorter).
fn resize_to(src: &[f32], target_width: usize) -> Vec<f32> {
    if target_width == 0 {
        return Vec::new();
    }
    let src_len = src.len();
    if src_len == 0 {
        return vec![0.0; target_width];
    }
    if src_len == target_width {
        return src.to_vec();
    }

    if src_len > target_width {
        // Average-pool each contiguous chunk into a single output value.
        let mut out = Vec::with_capacity(target_width);
        for i in 0..target_width {
            let start = (i * src_len) / target_width;
            let end = ((i + 1) * src_len) / target_width;
            let end = end.max(start + 1).min(src_len);
            let slice = &src[start..end];
            let mean = slice.iter().sum::<f32>() / slice.len() as f32;
            out.push(mean);
        }
        out
    } else {
        // Pad with zeros on the right (stable & deterministic).
        let mut out = Vec::with_capacity(target_width);
        out.extend_from_slice(src);
        out.resize(target_width, 0.0);
        out
    }
}

/// Apply `tanh` element-wise in place so every entry lies in `(-1, 1)`.
fn squash_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.tanh();
    }
}

/// Project a `cortex_tensor::Tensor` embedding into a bounded SNN stimulus
/// of length `embedding.dim()` (no resizing).
///
/// * 2-D `[seq, dim]` tensors are mean-pooled along the sequence axis.
/// * 1-D tensors are used directly.
/// * The final values are `tanh`-squashed into `(-1, 1)`.
pub fn embed_to_stimuli(embedding: &Tensor) -> Vec<f32> {
    let mut pooled = mean_pool(embedding);
    squash_inplace(&mut pooled);
    pooled
}

/// Project an embedding into a bounded SNN stimulus of length `snn_width`.
///
/// Pipeline (in order): **pool → resize → tanh**. The `tanh` is applied
/// last so the final vector is always in `(-1, 1)` regardless of how the
/// resize step combined the pooled values.
pub fn embed_to_stimuli_with_width(embedding: &Tensor, snn_width: usize) -> Vec<f32> {
    let pooled = mean_pool(embedding);
    let mut resized = resize_to(&pooled, snn_width);
    squash_inplace(&mut resized);
    resized
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_to_stimuli_1d_bounded() {
        let t = Tensor::from_vec(vec![100.0, -100.0, 0.0, 50.0], &[4]);
        let out = embed_to_stimuli(&t);
        assert_eq!(out.len(), 4);
        // tanh saturates to exactly ±1.0 in f32 for very large inputs, so we
        // require `abs <= 1.0` (the closed [-1, 1] range).
        for v in &out {
            assert!(v.abs() <= 1.0, "tanh must bound values in [-1, 1], got {v}");
        }
        // tanh(100) ≈ 1.0, tanh(-100) ≈ -1.0
        assert!(out[0] > 0.99);
        assert!(out[1] < -0.99);
        assert!((out[2]).abs() < 1e-6); // tanh(0) == 0
    }

    #[test]
    fn test_embed_to_stimuli_2d_mean_pool() {
        // shape [2, 3]: rows [1, 2, 3] and [3, 4, 5] → mean [2, 3, 4]
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0], &[2, 3]);
        let out = embed_to_stimuli(&t);
        assert_eq!(out.len(), 3);
        // Values should be tanh(2), tanh(3), tanh(4)
        assert!((out[0] - 2.0f32.tanh()).abs() < 1e-5);
        assert!((out[1] - 3.0f32.tanh()).abs() < 1e-5);
        assert!((out[2] - 4.0f32.tanh()).abs() < 1e-5);
    }

    #[test]
    fn test_embed_to_stimuli_with_width_downsamples() {
        // 8 → 4 with a clean boundary: pairs are averaged then tanh'd.
        let t = Tensor::from_vec(
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            &[8],
        );
        let out = embed_to_stimuli_with_width(&t, 4);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 0.0f32.tanh()).abs() < 1e-5);
        assert!((out[1] - 1.0f32.tanh()).abs() < 1e-5);
        assert!((out[2] - 2.0f32.tanh()).abs() < 1e-5);
        assert!((out[3] - 3.0f32.tanh()).abs() < 1e-5);
    }

    #[test]
    fn test_embed_to_stimuli_with_width_pads() {
        let t = Tensor::from_vec(vec![0.5, -0.5], &[2]);
        let out = embed_to_stimuli_with_width(&t, 5);
        assert_eq!(out.len(), 5);
        // First two carry the signal, last three are tanh(0) == 0.
        assert!((out[0] - 0.5f32.tanh()).abs() < 1e-5);
        assert!((out[1] - (-0.5f32).tanh()).abs() < 1e-5);
        for v in &out[2..] {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_embed_to_stimuli_with_width_strictly_bounded_after_resize() {
        // Huge magnitudes: resize should average them, tanh should still
        // clamp the final output to (-1, 1).
        let t = Tensor::from_vec(vec![500.0; 128], &[128]);
        let out = embed_to_stimuli_with_width(&t, 16);
        assert_eq!(out.len(), 16);
        for v in &out {
            assert!(v.abs() <= 1.0, "tanh bound violated: {v}");
            assert!(*v > 0.99); // tanh(500) saturates positive
        }
    }

    #[test]
    fn test_empty_width_returns_empty() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let out = embed_to_stimuli_with_width(&t, 0);
        assert!(out.is_empty());
    }
}
