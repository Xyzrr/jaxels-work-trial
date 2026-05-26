# SWE-Hero Dataset Row-Count Discrepancy

Investigated: 2026-05-21

## Summary

The exact 13.2k-row SWE-HERO paper training set cannot be recovered from the
public Hugging Face artifact alone.

What the evidence shows:

- The paper reports about 13.2k execution-based trajectories generated as one
  rollout per task instance.
- The public `nvidia/SWE-Hero-openhands-trajectories` dataset observed during
  the investigation exposed 34,269 rows over 11,766 unique `instance_id` values.
- Every public data-bearing revision is a 32k-36k row multi-rollout pool. None
  has about 13.2k rows or about 13.2k unique instances.
- The local training approximation is
  `datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout/`: the earliest
  public upload reduced to one rollout per task, then context-capped to 12,617
  rows.

For shared ML vocabulary such as SFT, rollout, context window, and one rollout
per task, see [`../AGENTS.md`](../AGENTS.md).

## Why It Matters

This is a training-data weighting issue, not a cosmetic row-count mismatch.
Training on all public rows would overweight tasks with multiple attempts. A
synthetic 13.2k slice, such as the first 13,200 rows, would also preserve
duplicate rollouts and change the source/repo distribution.

The only faithful fix is a row-level paper manifest from the authors.

## Paper Claims

Source: arXiv `2604.01496`, v2 dated 2026-05-06.

Relevant claims:

- SWE-HERO is described as about 13k trajectories across about 13k task
  instances.
- 13.5k instances had containerized Docker environments and verified reference
  patches.
- The authors generated a single rollout per task instance.
- The final SWE-HERO set is 13.2k execution-based trajectories after filtering.
- SWE-HERO trajectories were not excluded based on task-resolution success.

## Public Dataset Evidence

Dataset: `nvidia/SWE-Hero-openhands-trajectories`

Observed public main revision:

- SHA: `150bc119e52c647216fce285fd801f16b6fd745b`
- Last modified: 2026-05-08T17:10:16Z
- Public refs: `main` and `refs/convert/parquet`; no public tags or alternative
  branches were present via `git ls-remote`.
- Dataset card: 11,766 total issues, 34,269 total trajectories.
- Sources: `SWE-Gym/SWE-Gym`, `R2E-Gym/R2E-Gym-Subset`,
  `nebius/SWE-rebench`

Direct parquet query of that revision:

| Metric | Count |
| --- | ---: |
| Rows | 34,269 |
| Unique `instance_id` values | 11,766 |
| Instances with 1 trajectory | 329 |
| Instances with 2 trajectories | 371 |
| Instances with 3 trajectories | 11,066 |

Source breakdown:

| `dataset` value | Rows | Unique instances | Unique repos |
| --- | ---: | ---: | ---: |
| `nebius/SWE-rebench` | 18,189 | 6,261 | 1,688 |
| `R2E-Gym/R2E-Gym-Subset` | 10,221 | 3,442 | 8 |
| `SWE-Gym/SWE-Gym-Raw` | 3,990 | 1,397 | 10 |
| `SWE-Gym/SWE-Gym` | 1,869 | 666 | 1 |

License breakdown:

| License | Rows | Unique instances |
| --- | ---: | ---: |
| `BSD-3-Clause` | 13,047 | 4,465 |
| `MIT` | 11,208 | 3,856 |
| `Apache-2.0` | 8,914 | 3,063 |
| `BSD-2-Clause` | 1,094 | 380 |
| `MIT-0` | 6 | 2 |

## Public Revision History

The dataset repo was created after the v1 paper submission:

- Paper v1 submission: 2026-04-02
- HF dataset initial commit: 2026-04-17
- First data upload: 2026-04-20

Historical parquet object counts from public `main` history:

| Date | Commit | Files | Rows | Unique instances | Instances by trajectory count |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-04-20 | `5b2ed21` | 15 | 35,934 | 12,633 | 1:633, 2:699, 3:11,301 |
| 2026-04-20 | `d294bb6` | 15 | 35,934 | 12,633 | 1:633, 2:699, 3:11,301 |
| 2026-04-20 | `17c8d28` | 13 | 32,368 | 11,409 | 1:601, 2:657, 3:10,151 |
| 2026-05-01 | `fdabc1a` | 14 | 34,386 | 11,806 | 1:330, 2:372, 3:11,104 |
| 2026-05-01 | `7ec4ffc` | 14 | 34,269 | 11,766 | 1:329, 2:371, 3:11,066 |
| 2026-05-08 | `150bc11` | 14 | 34,269 | 11,766 | 1:329, 2:371, 3:11,066 |

None of these revisions contains about 13.2k rows or 13.2k unique instances.
The closest one-trajectory-per-instance candidate is the earliest upload at
12,633 unique instances, still 567 short of 13.2k.

The May 1 morning upload (`fdabc1a`) also appears to have a different
`dataset` column convention: grouping by `dataset` yields repository names
rather than the source labels used in current main. That suggests the public
artifact was still being cleaned after the paper, not frozen as the paper SFT
manifest.

## Interpretation

The current public dataset is best understood as a multi-rollout public pool
over about 11.8k task instances, not the final 13.2k paper training set.

Most plausible sequence:

1. Internal paper run: one rollout per Docker-backed task, filtered to about
   13.2k trajectories.
2. Public release: a broader multi-rollout trajectory pool was exported after
   the paper submission.
3. Public follow-up edits changed counts, license filtering, source breakdown,
   schema/card details, and at least one column convention.

The public rows lack enough provenance to reconstruct the exact paper subset:
rollout ordinal/seed, exact source revisions, task whitelist,
reference-patch-verification result, filter flags, test-patch metadata, and
final SFT manifest are missing.

## Local Approximation

Use this artifact only with an explicit caveat that it is not the exact paper
training set.

Raw one-rollout approximation:

- Script: `scripts/prepare_swehero_historical_one_rollout.py`
- Source revision: `5b2ed21270ad773a50163e2999c510f0cbb92cfa`
- Output: `datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout/`
- 2026-05-21 shape: 12,633 selected rows from 35,934 public rows; 12,633 unique
  instances.
- Training path: pass the output directory as `--dataset-id`.

The script applies paper filters visible in public columns and records the
missing `test_patch`-overlap filter caveat in generated `metadata.json`.

Context-capped training artifact:

- Script: `scripts/refresh_swehero_context_capped_one_rollout.py`
- Input/output: `datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout/`
- Tokenization: Qwen2.5-Coder ChatML over OpenHands messages, matching
  `scripts/qwen_swehero_train.py`; model patches are not appended by default.
- Context cap: shifted input length must be `<= 131,072` tokens.
- Replacement rule: for over-context selected rows, choose a fitting same-task
  rollout by lowest `str_replace_editor` error count, then lowest assistant turn
  count, then earliest source row index.
- 2026-05-22 shape: 12,617 selected rows. Of 39 over-context selected rows, 23
  were replaced and 16 tasks were excluded.
- Verification: 12,617 rows, 12,617 unique `instance_id` values, zero
  over-context rows, max shifted input length 130,126, and zero manifest length
  mismatches.
- Audit file: `context_filter_report.json`

The training workflow consumes this artifact as documented in
[`../docs/swehero_torchtitan_pod.md`](../docs/swehero_torchtitan_pod.md).

## Public Approximation Options

| Option | Revision | One-rollout result | Use when | Caveat |
| --- | --- | ---: | --- | --- |
| Current public main | `150bc119e52c647216fce285fd801f16b6fd745b` | 11,766 | Matching the current dataset card and license cleanup matters most. | Materially smaller than the paper set. |
| Earliest public upload | `5b2ed21270ad773a50163e2999c510f0cbb92cfa` | 12,633 | Closest public unique-instance count matters most. | Still not 13.2k; early post-paper upload with broader/noisier license labels. |

A deterministic tie-breaker is required for either option. Without one, two
rebuilds from the same public pool could pick different rollouts for the same
task and produce different SFT targets.

## Excluded Context-Refresh Tasks

These 16 tasks had no accepted same-task rollout that fit the 131,072-token
shifted-input cap during the 2026-05-22 refresh:

```text
juanifioren__django-oidc-provider-329
matthewwithanm__django-imagekit-574
nedbat__coveragepy-0d6449874cd4d3003ce908d66fa654b64bfea0c0
nedbat__coveragepy-1cd6c9bba0b4ba3018bf1b28fee645a7dd98fe68
nedbat__coveragepy-35e249ff74cfcbc44889107cfcca785696dc4288
nedbat__coveragepy-423fa596325acb8f6bcb37a3502cf7853e5d395a
nedbat__coveragepy-84f70f69c5e3f7117d219f842ef66ec037478bc9
nedbat__coveragepy-8eb95b5ad2ed1cee1204b1ce95bad9118063d178
nedbat__coveragepy-9209c555c7612b4a649edca5db97a04177ee5a9a
nedbat__coveragepy-d723b46460dc7ffb4abf54806087ffd614b81331
nedbat__coveragepy-df1bf082f242cccdcb342000525bede537b95935
nedbat__coveragepy-ff2b70a39bbe5f6b6e1752e4664fad64211d2280
nipy__nipype-2669
python__mypy-11125
python__mypy-11521
streamlink__streamlink-3131
```

## Request To Authors

If exact reproduction is required, ask NVIDIA for the row-level training
manifest rather than another natural-language clarification.

Suggested request:

```xml
<request>
  <context>
    We are trying to reproduce the SWE-HERO SFT stage from arXiv:2604.01496.
    The paper reports 13.2k execution-based trajectories, generated as one rollout
    per task instance after filtering. The public dataset
    nvidia/SWE-Hero-openhands-trajectories currently contains 34,269 rows over
    11,766 unique instance_id values, with almost all instances having 3 trajectories.
  </context>
  <ask>
    Please publish or share the exact row-level manifest used for the paper's
    SWE-HERO SFT run, preferably as instance_id plus trajectory_id, with the
    source dataset revisions and filtering criteria.
  </ask>
  <why>
    The public Hugging Face revision history does not contain a 13.2k-row artifact
    or a 13.2k-unique-instance artifact, so the exact paper training subset cannot
    be reconstructed from public metadata.
  </why>
</request>
```

## Reproduction Commands

Historical refs:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories /tmp/swe-hero-hf
git -C /tmp/swe-hero-hf log --oneline --date=iso --format='%h %ad %an %s' --all
git -C /tmp/swe-hero-hf ls-remote https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories
```

Current parquet counts:

```bash
uv run --with duckdb --with requests python - <<'PY'
import duckdb, requests
repo = "nvidia/SWE-Hero-openhands-trajectories"
urls = requests.get(f"https://huggingface.co/api/datasets/{repo}/parquet").json()["default"]["train"]
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
print(con.execute("select count(*), count(distinct instance_id) from read_parquet(?)", [urls]).fetchone())
print(con.execute("""
with c as (
  select instance_id, count(*) n
  from read_parquet(?)
  group by instance_id
)
select n, count(*) from c group by n order by n
""", [urls]).fetchall())
PY
```

Historical parquet counts:

```bash
uv run --with duckdb python - <<'PY'
import duckdb, subprocess
repo = "nvidia/SWE-Hero-openhands-trajectories"
commits = [
    "5b2ed21270ad773a50163e2999c510f0cbb92cfa",
    "d294bb6a41d7f8d9791b00001c5fb7f884e78352",
    "17c8d28f6bcf2ae9578024d7dd668778979e3e0f",
    "fdabc1a24f1e5ba574b0501d8121edfaf70ffbf8",
    "7ec4ffcc57c1fbf038c860a5e2a62045ec5f50ea",
    "150bc119e52c647216fce285fd801f16b6fd745b",
]
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
for c in commits:
    names = subprocess.check_output(
        ["git", "-C", "/tmp/swe-hero-hf", "ls-tree", "-r", "--name-only", c, "data"],
        text=True,
    ).splitlines()
    urls = [f"https://huggingface.co/datasets/{repo}/resolve/{c}/{name}" for name in names if name.endswith(".parquet")]
    row_count, instance_count = con.execute(
        "select count(*), count(distinct instance_id) from read_parquet(?)",
        [urls],
    ).fetchone()
    dist = con.execute("""
    with x as (
      select instance_id, count(*) n
      from read_parquet(?)
      group by instance_id
    )
    select n, count(*) from x group by n order by n
    """, [urls]).fetchall()
    print(c[:7], len(urls), row_count, instance_count, dist)
PY
```
