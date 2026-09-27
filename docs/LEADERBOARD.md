# Leaderboard submission and audit process

This is the process a result goes through before it is listed. It exists so that
every listed number can be checked by someone other than the person who produced
it. The tooling is in the repository; the leaderboard itself opens when a shot
bank is marked `leaderboard_eligible` (none is yet — every bank in v0.3.0 is
`experimental`).

## 1. What a submitter provides

A result package built by `scripts/package_submission.py` from a **clean
checkout** of a tagged release:

```bash
python scripts/package_submission.py --policy my_pkg:load_policy \
    --policy-id my-policy-v1 --split test --levels all \
    --policy-file weights/actor.pt --out submission/
```

The package holds the manifest (commit, task, bank digests, backend, hardware,
reproduce command), the full task configuration and run arguments, every
episode's raw result per level, the weights with their sha256, fixed success
*and* failure videos, and the environment's package versions. See
[`SUBMISSION.md`](SUBMISSION.md) for the format and
[`schemas/submission.v0.json`](../src/multisport_sim/benchmark/schemas/submission.v0.json)
for its schema.

Requirements for listing:

| Requirement | Why |
|---|---|
| `repository.dirty` is `false` | a dirty tree cannot be reproduced from a commit |
| every test level run in full | partial runs select their own episodes |
| weights included (`policy.complete`) | without them nobody else can re-run it |
| the bank is `leaderboard_eligible` | experimental banks may still change |
| training seeds and configuration in `config/` | BENCHMARK_SPEC §8 asks for 5 seeds |

## 2. What the maintainer runs

```bash
python scripts/audit_submission.py submission/ --report audit.json          # static checks
python scripts/audit_submission.py submission/ --rerun --report audit.json  # plus replay
```

| Check | Hard? | What it catches |
|---|:---:|---|
| `structure` | yes | missing manifest, metrics, config, environment or weight hashes |
| `integrity.shot_bank` | yes | a package scored on a bank this release does not ship |
| `<level>.completeness` | yes | dropped or duplicated episodes: the reported shot ids must be exactly the bank's |
| `<level>.verdict` | yes | a pass/fail or primary value that does not follow from the raw episodes |
| `<level>.metrics` | yes | any aggregate metric that does not follow from the raw episodes |
| `integrity.policy_files` | yes | weights that do not match their recorded sha256 |
| `<level>.replay` (`--rerun`) | yes | episodes that come out differently when the reproduce command is executed |
| `provenance.clean_tree` | warning | a package built from a dirty tree |
| `provenance.weights_included` | warning | a package without weights |
| `eligibility.leaderboard` | warning | an experimental bank |

The static checks recompute everything from the package's own evidence with the
frozen criteria in the shipped bank manifest; they do not trust a single
summary number in `metrics.json`. The replay check runs on the maintainer's
machine: MuJoCo is deterministic on one machine, so any difference there is a
finding. Across machines, floating-point differences can flip individual
episodes; the maintainer states the hardware in the audit record.

## 3. Listing

A result is listed with: the audit record (`audit.json`), the package's commit
and bank digest, the track (`state` or `vision`) and whether the policy reads
privileged state. Results are never merged across tracks, banks or releases.

## 4. What is not decided yet

| Item | Status |
|---|---|
| Hidden test set | The test split is public; a held-out set for listed results is planned (M5) |
| Hosting | Where the table lives and who holds write access |
| Appeals | How a disputed audit is re-examined |
| External reproduction | M6 asks for at least one external user to reproduce a result; none has yet |
