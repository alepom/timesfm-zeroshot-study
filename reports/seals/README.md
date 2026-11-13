# Seals and public timestamps

- `SEAL_*.json`, `ISSUE_*.json`: Tier P seals written by `python -m tfmstudy.data.prospective seal` (never re-run: it overwrites).
- `TIMESTAMP_MANIFEST_<date>.txt`: plain-text list of the hashes to be timestamped (protocol commit, protocol file, combined seal,
  seal and issue files). Only the manifest's sha256 left the machine.
- `*.tsq` / `*.tsr`: RFC 3161 request and reply from FreeTSA (https://freetsa.org); certificates in `freetsa/`. Verify with
  `openssl ts -verify -in X.tsr -data X.txt -CAfile freetsa/cacert.pem -untrusted freetsa/tsa.crt`
  and read the time with `openssl ts -reply -in X.tsr -text`.
- `*.txt.ots`: OpenTimestamps proof (https://opentimestamps.org). Verify with `ots verify X.txt.ots` (needs the Bitcoin
  attestation, added by `ots upgrade X.txt.ots` a few hours after stamping) or by dropping both files on opentimestamps.org.

| Manifest | Contents | FreeTSA time (UTC) | OpenTimestamps |
|---|---|---|---|
| 2026-11-02 | issue 2026-11-02 seal (0d96ab18…) | 2026-11-02 10:28:11 | Bitcoin block 974245 (2026-11-02 10:42:58 UTC); also 974257 |
| 2026-10-30 | protocol-frozen-v1 (9f15b38…), issue 2026-10-30 seal (f84a669d…) | 2026-10-30 13:29:45 | Bitcoin block 973830 (2026-10-30 13:48:06 UTC); also 973864, 973887 |

Without a local Bitcoin node, `ots verify` cannot run; the proof was checked on 2026-10-31 by recomputing each attestation's
merkle root with the `opentimestamps` library and comparing it with the block header from blockstream.info (all three match).

## History rewrite before release (2026-11-13)

Before the repository was published its history was rewritten to replace a private contact address (commit metadata and the
User-Agent line of `tfmstudy/data/sources.py`) and to remove internal planning notes. Every commit hash changed. The 2026-10-30
manifest therefore names the freeze commit by its earlier hash, `9f15b38…`; the tag `protocol-frozen-v1` now points to `ce1c1e5`.
What remains verifiable:

- `protocol.md` at the tag has the SHA-256 listed in the 2026-10-30 manifest (`45d7078f…`):
  `git show protocol-frozen-v1:protocol.md | sha256sum`.
- Each Tier P seal record lists the SHA-256 of every sealed file and the combined hash is the SHA-256 of those `name:digest` lines,
  so the combined hash in each manifest can be recomputed from the seal record, and every released file (contexts, tasks, issue
  record, run records, forecasts; Zenodo data record) can be checked against its digest.
- The code bundle listed in the seals (`tfmstudy-0.5.4-…zip`) contains the same private address and is not released; its digest
  cannot be checked by third parties.

