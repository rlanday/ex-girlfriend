# Fast-weight auxiliary head in Ex-Girlfriend 1.0

This describes the original FINAL implementation. Source paths are relative to the repository root.

Let `x ∈ R^192` be the unmodified final transformer residual and `h_b ∈ R^768`, for `b=0..3`, the feature vector from MLP blocks 12..15 (zero-based). The implementation reads the **quantized ReLU² activation code**, multiplied by that block's fixed activation scale. Dead neurons already omitted from the shipped model remain absent; zero activations are skipped. Each matrix `ΔW_b ∈ R^(192×768)` starts at zero and persists across documents. It is runtime state reconstructed by both coder sides, not an additional shipped checkpoint.

Forward rule:

```
δx = Σ_b ΔW_b h_b
z  = x + δx
p_FW = softmax(15 tanh(U Q(RMSNorm(z)) / 15))
```

Here `Q` and the output projection are the existing quantized native head. The actual head includes its inherited projection details; this equation summarizes the dataflow. The original residual, original head scratch, static probabilities and static hidden-state accessor are left untouched by the added readout. All four corrections enter **one final auxiliary head**; they are not injected back into intermediate layers. See `src/tf1/opt/model_opt.cpp`.

The auxiliary update differentiates its own softmax cross-entropy through the soft cap and a float reconstruction of the frozen unembedding, then uses an RMS-normalization backward rule:

```
d_o = (p_FW[o] − 1[o=y]) (1 − (capped_logit[o]/15)^2)
a   = U_float^T d
g   = r (a − z_norm mean(a ⊙ z_norm))
```

`r` is recovered as `sqrt(sum(z_norm²)/sum(z²))`, or zero when the denominator is zero. Because the forward path quantizes activations, this is a **straight-through-style surrogate update**, not the exact derivative of the quantized encoder. There is no gradient through frozen feature extraction or the host mixer/SSE/BITLSTM. The surrogate is an implementation choice; it is not an exact gradient of the final mixed compression loss.

The FWRH update uses per-feature squared-activation averages, per-block normalization and one shared denominator:

```
v_bi ← 0.999 v_bi + 0.001 h_bi²
m_b  = mean_i sqrt(v_bi)
sh_bi = h_bi m_b / (sqrt(v_bi) + 0.001 m_b + 1e−12)
κ_t  = 0.0015 + 0.0045 / (1 + t/8000)
ΔW_b[:,i] ← ΔW_b[:,i] − κ_t sh_bi g / (1 + Σ_b,i h_bi sh_bi)
```

The code decays all 768 statistics in each block but writes matrix columns only for nonzero features. `t` counts eligible updates, not every raw input byte, and neither it nor the matrices reset per article. The epsilon and unit denominator make the zero-feature case well defined. Exact implementation: `src/tf1/opt/model_opt.cpp:651–713`.

**Causal pairing matters.** The engine updates at the beginning of the next stream step, after the target byte is known, only if `fw_have && fw_live && opt`. A read row may be a prompt row; unread intermediate rows must not replace the stored feature/gradient scratch. The target is the next stream symbol past inserted rows, including the closing symbol of a document. The engine tracks when the actual slot row came from the auxiliary head rather than PPMd, neutrality or abstention. See `src/tf1/tf1_engine.cpp:254–279,473–477,729–746`. “An update after every coded byte” would overstate this contract.

The auxiliary distribution undergoes the same float16 round trip and `1e−6` floor as the static transformer row. A separate byte-slot object converts it into the current bit's conditional input. `PORT_GEOM` adds layer-0 input 588 (589 inputs total); **FW1** also supplies its stretched value directly to final-mixer input 27 (28 inputs total). Adaptive-depth gather and BITLSTM input shape remain the inherited ones. Gated metadata, alias feeding and timestamp skip paths neutralize both slots. See `src/predictor.cpp:427–431,469–475,579–580,655–707`.

The new matrices occupy 2,359,296 bytes (2.25 MiB), plus 159,744 bytes for the float unembedding copy and feature/mixer scratch. Runtime state is learned from decoded history and is not stored as a new pretrained asset. See [ABLATIONS.md](ABLATIONS.md) for measured gains and costs.
