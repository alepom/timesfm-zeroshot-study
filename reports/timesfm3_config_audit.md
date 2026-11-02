# TimesFM-3 `config.json` audit (mechanism report §8, E10) — 2026-10-30

`google/timesfm-3.0-pytorch`, revision `43046b85ec22d584a13f8098c2ed39c889e129c2` (the revision recorded in every run file),
`config.json` sha256 `ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6`. The file is not copied into this
repository (it belongs to the model repository, non-commercial licence); the values that settle the open points of
`reports/mechanism.md` are:

| Field | Value | Consequence |
|---|---|---|
| `use_rope_var` | `false` | variate attention has no positional signal: permutation-equivariant over variates; a past-only covariate and an extra target are indistinguishable inside the network (mechanism §2.2, §4.3) |
| `causal_attention` | `true` | sequence attention is causal (decoder-only along time) |
| `use_rope_seq` | `true` | RoPE on the patch axis |
| `input_transform` | `"identity"` | the unused `input_transform` code path is harmless |
| `use_frozen_running_stats` | `false` | statistics of past-future covariates keep updating with their future values |
| `use_linear_detrending`, `linear_detrending_threshold` | `true`, `0.5` | detrending gate std ratio 0.5 (R² > 0.75), as in the code default |
| `use_iterative_cpm_revin`, `use_stitching`, `use_variate_attention` | `true` | as the code defaults |
| `max_variates`, `paired_token_skip_second` | `32`, `false` | declared but not read by the inference code |
| `num_layers`, `model_dims`, `num_heads`, FFN `hidden_dims` | 20, 1280, 16, 1280 | as in mechanism §4.6 (≈ 330.7M parameters) |
| `training`, `dropout` | `true`, `0.0` | `training` is a stored flag; with dropout 0 and `model.eval()` it has no effect at inference |

## Weights history (checked 2026-11-02, Hugging Face API, read-only)

The revision used, `43046b85…` (commit of 2026-09-02, "Test removing timesfm tag to check library_name"), is later than the
start of the Tier B test windows (2026-09-01). The file list of every revision shows that `model.safetensors` (1,322,898,824
bytes, LFS sha256 `a7592b0a8432baee…`) and `config.json` (git blob `c1de41214b9c…`) are identical from the first upload
(`6455822fd470`, 2026-08-24 22:41 UTC, "Upload TimesFM 3.0 PyTorch model weights and config") to the revision used; the eight
later commits change only README, LICENSE and metadata. The network evaluated is therefore the one published before the
test windows began.
