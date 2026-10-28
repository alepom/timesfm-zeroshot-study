# TiRex-2 checkpoints: provenance check (2026-10-28)

Evidence file: `data/manifests/tirex2_provenance.json` (Hugging Face API snapshot, tensor comparison, forecast check).

## What was checked

1. **Forecasts.** On the GPU test (Colab T4, tirex-2 0.4.0, fev tasks ETT_1D and epf_de), `NX-AI/TiRex-2`
   (revision `30788a31`) and `NX-AI/TiRex-2-fevbench` (revision `acd6153b`) gave bit-identical quantile forecasts.
2. **Weights.** `model.safetensors` of both repositories downloaded; local sha256 equal to the Hub's LFS oid (download
   intact). All 283 tensors are byte-identical. The two files differ only in their header metadata (the main one has an
   extra `source` field, hence 24 bytes more); both record the same `source_sha256` = `184b160f…fcac1`, the sha256 of
   the `model.ckpt` they were converted from. `TiRex-2-gifteval-zs`, used as a negative control, differs in all 283 tensors.
3. **Configuration and loader.** `config.json` and `model-config.yaml` are identical in the two repositories. The
   tirex-2 0.4.0 loader reads `config.json` + `model.safetensors` (falls back to the legacy files only if missing).
4. **History** (commit dates in UTC, from the Hub API):

| Repository | Period | `model.ckpt` sha256 |
|---|---|---|
| TiRex-2 | 2026-06-16 → 2026-07-01 (commits `49b3f7e1` … `35bde0d6`) | `1ab4b01a…` |
| TiRex-2-fevbench | first weights 2026-07-02 04:48:15 (`2ab379b2`) | `184b160f…` |
| TiRex-2 | from 2026-07-02 04:48:46 (`f500c26e`, "Upload folder using huggingface_hub") | `184b160f…` |
| TiRex-2-gifteval-zs | since 2026-06-25 | `af3ad356…` |
| TiRex-2-gifteval-pretrain | since 2026-06-25 | `187b4ab4…` |

   The `1ab4b01a…` checkpoint published as TiRex-2 until 2026-07-01 does not appear in any other repository.
5. **Documentation.** The TiRex-2 model card (then and now) presents TiRex-2-f as a separate checkpoint trained without
   the fev-bench evaluation datasets. The paper (arXiv:2607.01204v1, 1 July 2026, §4.1 and Appendix E) states that one
   checkpoint was pretrained per benchmark, with overlapping datasets removed, and that each benchmark's scores come from
   the corresponding checkpoint; it does not say which checkpoint is published as `NX-AI/TiRex-2`. No discussion on the
   Hub repositories mentions the point (checked 2026-10-28).

## Reading

The facts are compatible with two explanations that public material cannot distinguish: (a) since 2026-07-02 the main
repository intentionally hosts the fev-bench-decontaminated checkpoint, or (b) the same file was uploaded to both
repositories by mistake. No claim about intent is made.

## Consequences for this study

- The planned TiRex-2 vs TiRex-2-f contamination contrast in Tier A is dropped (protocol §3.1, before the freeze).
- All TiRex-2 results use revision `30788a31` (weights `184b160f…`, recorded in every run file). By the TiRex-2-f model
  card, this checkpoint was trained without the fev-bench evaluation datasets, which would make TiRex-2's Tier A scores
  zero-shot in the strict sense; we state this as conditional on the card.
- Separately (not a finding, for the record): TiRex-2-gifteval-zs also differs from TiRex-2 in configuration
  (context 8,192 vs 2,048, output 928 vs 320, no arcsinh scaling, no QK norm), so it is not a same-architecture variant.
  It is not used in the study.

## Draft note for the paper (neutral wording)

> *Note on TiRex-2 checkpoints.* At the time of our experiments (October 2026), the weights published as
> `NX-AI/TiRex-2-fevbench`, described as excluding the fev-bench evaluation datasets from pretraining, were identical
> to those published as `NX-AI/TiRex-2`: all 283 tensors matched byte for byte, both files derive from the same
> checkpoint (sha256 `184b160f…`), and the two models produced bit-identical forecasts. The repository history shows
> this checkpoint uploaded to both repositories on 2 July 2026, where it replaced the checkpoint previously published
> as TiRex-2. We therefore could not use the two variants to estimate the effect of fev-bench data in pretraining, and
> our TiRex-2 results refer to this checkpoint (revision `30788a31`). Hashes and commit identifiers are in the
> replication package.
