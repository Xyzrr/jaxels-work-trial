# SWE-Hero Dataset Row-Count Discrepancy

Date investigated: 2026-05-21

## Question

The SWE-ZERO to SWE-HERO paper reports a SWE-HERO training set of 13.2k execution-based trajectories, but the linked Hugging Face dataset currently exposes 34,269 rows. We need to know whether the exact 13.2k rows used in the paper can be recovered from the public artifact.

## Reader Vocabulary

These terms determine why the row counts below matter for training:

- A task instance, stored as `instance_id`, is one coding problem plus its
  repository/test environment. If the same `instance_id` appears three times,
  the model would see three attempts at the same problem, not three independent
  problems.
- A rollout or trajectory is one full OpenHands attempt at solving one task:
  prompts, assistant actions, tool calls, tool observations, and usually a
  produced patch. Supervised fine-tuning (SFT) turns the assistant portions of
  those traces into next-token targets.
- Execution-based means the data came from an agent interacting with an actual
  repository environment, rather than from a hand-written instruction/answer
  pair. That makes the traces valuable, but also makes provenance important
  because tool failures, duplicate attempts, and over-long contexts change what
  the model learns.
- One rollout per task is an ML weighting decision. It keeps every retained
  task roughly equally represented. Training on all public rollouts would
  over-weight tasks with multiple attempts and would no longer match the paper's
  described data distribution.

## Paper Claims

Source: arXiv `2604.01496`, v2 dated 2026-05-06.

Relevant claims:

- The SWE-HERO stage is described as a refined collection of about 13k trajectories across about 13k task instances.
- The task collection section says 13.5k instances had containerized Docker environments and were verified by executing reference patches.
- The SWE-HERO trajectory collection says the authors generated a single rollout per task instance.
- The corpus composition section says the final SWE-HERO set comprises 13.2k execution-based trajectories after the filtering protocol.
- The same paragraph says they did not exclude SWE-HERO trajectories based on task-resolution success because the SWE-HERO dataset is smaller.

This implies the paper training set should be approximately one trajectory per retained task instance, not three trajectories for most task instances. That difference is not cosmetic: duplicate trajectories for a task change the effective training weights and can teach the model to imitate repeated attempts at the same issue.

## Public Hugging Face Dataset

Dataset: `nvidia/SWE-Hero-openhands-trajectories`

Current main revision observed:

- SHA: `150bc119e52c647216fce285fd801f16b6fd745b`
- Last modified: 2026-05-08T17:10:16Z
- Public refs: `main` and `refs/convert/parquet`; no tags or alternative public branches were present via `git ls-remote`.
- Dataset card reports:
  - Total issues: 11,766
  - Total trajectories: 34,269
  - Sources: `SWE-Gym/SWE-Gym`, `R2E-Gym/R2E-Gym-Subset`, `nebius/SWE-rebench`

Direct parquet query of current main confirms:

| Metric | Count |
| --- | ---: |
| Rows | 34,269 |
| Unique `instance_id` values | 11,766 |
| Instances with 1 trajectory | 329 |
| Instances with 2 trajectories | 371 |
| Instances with 3 trajectories | 11,066 |

Current source breakdown:

| `dataset` value | Rows | Unique instances | Unique repos |
| --- | ---: | ---: | ---: |
| `nebius/SWE-rebench` | 18,189 | 6,261 | 1,688 |
| `R2E-Gym/R2E-Gym-Subset` | 10,221 | 3,442 | 8 |
| `SWE-Gym/SWE-Gym-Raw` | 3,990 | 1,397 | 10 |
| `SWE-Gym/SWE-Gym` | 1,869 | 666 | 1 |

Current license breakdown:

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

Historical parquet object counts from the public `main` history:

| Date | Commit | Files | Rows | Unique instances | Instances by trajectory count |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-04-20 | `5b2ed21` | 15 | 35,934 | 12,633 | 1:633, 2:699, 3:11,301 |
| 2026-04-20 | `d294bb6` | 15 | 35,934 | 12,633 | 1:633, 2:699, 3:11,301 |
| 2026-04-20 | `17c8d28` | 13 | 32,368 | 11,409 | 1:601, 2:657, 3:10,151 |
| 2026-05-01 | `fdabc1a` | 14 | 34,386 | 11,806 | 1:330, 2:372, 3:11,104 |
| 2026-05-01 | `7ec4ffc` | 14 | 34,269 | 11,766 | 1:329, 2:371, 3:11,066 |
| 2026-05-08 | `150bc11` | 14 | 34,269 | 11,766 | 1:329, 2:371, 3:11,066 |

None of the public data-bearing revisions contains about 13.2k rows. None contains about 13.2k unique instances either. The closest public one-trajectory-per-instance candidate is the earliest upload at 12,633 unique instances, still 567 instances short of 13.2k and with a broader/noisier license set than the current card.

The May 1 morning upload (`fdabc1a`) also appears to have a different or incorrect `dataset` column convention: grouping by `dataset` yields repository names rather than the source labels used in current main. That is another signal that the public artifact was being iterated after the paper, not a frozen paper-training manifest.

## Likely Explanation

The current 34.2k-row public dataset is best understood as a multi-rollout public pool over 11.8k task instances, not as the final 13.2k paper training set.

Reasons:

1. The paper says SWE-HERO generated a single rollout per task instance, but current public main has three trajectories for 11,066 of 11,766 instances.
2. The paper says 13.5k Docker-backed instances were the SWE-HERO foundation and 13.2k trajectories remained after filtering. Public main has only 11,766 unique instances.
3. The dataset repo did not exist publicly until after the paper's v1 submission, and every public data upload is a 32k-36k row multi-rollout corpus.
4. Public revisions show active post-paper changes in counts, license filtering, source breakdown, and at least one column convention.
5. The public rows do not include enough provenance to reverse-engineer the paper's exact selection, such as rollout ordinal/seed, exact source dataset revisions, task whitelist, reference-patch verification result, filter flags, test-patch metadata, or a final SFT manifest.

The most plausible story is:

- Internal paper run: one rollout per Docker-backed task, filtered to about 13.2k trajectories.
- Public release: later exported a broader multi-rollout trajectory pool, then further edited it for licensing/source/schema/card cleanup.

## Can We Recover The Exact 13.2k Rows?

Not from the current public information.

The exact paper set requires a row-level manifest from the authors, ideally keyed by:

- `instance_id`
- `trajectory_id`
- source dataset and source revision
- OpenHands/version or trajectory-generation revision
- filtering decisions

Without that, any 13.2k subset we create from the 34,269 public rows would be invented. In particular, taking the first 13,200 rows is invalid because it preserves duplicate rollouts for early instances and changes the source/repo distribution.

## Best Public Approximation Options

Use one of these only with an explicit caveat that it is not the exact paper training set.

A deterministic tie-breaker is required for either approximation. Without one,
two rebuilds from the same public pool could pick different rollouts for the
same task and produce different SFT targets.

1. Current canonical public approximation:
   - Pin revision `150bc119e52c647216fce285fd801f16b6fd745b`.
   - Select one trajectory per `instance_id` with a declared deterministic tie-breaker.
   - Result: 11,766 trajectories.
   - Pros: matches the current public dataset card and license cleanup.
   - Cons: materially smaller than the paper's 13.2k and cannot reproduce the paper training set.

2. Largest historical public unique-instance approximation:
   - Pin revision `5b2ed21270ad773a50163e2999c510f0cbb92cfa`.
   - Select one trajectory per `instance_id` with a declared deterministic tie-breaker.
   - Result: 12,633 trajectories.
   - Pros: closest public unique-instance count.
   - Cons: still not 13.2k, uses an early post-paper upload, and includes license labels later removed from the public card/current data.

Implemented local preparation workflow:

- Script: `scripts/prepare_swehero_historical_one_rollout.py`
- Default output: `datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout/`
- Source revision: `5b2ed21270ad773a50163e2999c510f0cbb92cfa`
- Output shape from the 2026-05-21 run: 12,633 selected rows from 35,934 public rows and 12,633 unique instances.
- Training path: pass the output directory as `--dataset-id`.

The script applies the paper filters observable from public columns and records
the missing `test_patch`-overlap filter caveat in the generated `metadata.json`.

Implemented context-capped training artifact:

The context-capped refresh exists because long agent traces can exceed the
model's training window. In causal language-model training, the model reads a
sequence of tokens and predicts the next token; after that one-token shift, the
input length still must fit the configured context window. Rows that exceed the
131,072-token shifted-input cap would either fail training or require
truncation. Truncation is not used here because it can remove earlier tool
observations or the final assistant action, changing the supervised target.

- Script: `scripts/refresh_swehero_context_capped_one_rollout.py`
- Default input/output: `datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout/`
- Tokenization contract: Qwen2.5-Coder ChatML over OpenHands messages, matching `scripts/qwen_swehero_train.py`; model patches are not appended by default.
- Context cap: shifted input length must be `<= 131,072` tokens, equivalent to the 128k paper training context.
- Selection rule for over-context rows: among same-task rollouts that pass the same public-column filters and fit the context cap, choose the lowest `str_replace_editor` error count, then lowest assistant turn count, then earliest source row index.
- Output shape from the 2026-05-22 refresh: 12,617 selected rows. The raw artifact had 39 over-context selected rows; 23 were replaced with fitting same-task rollouts and 16 tasks were excluded because no accepted same-task rollout fit the context cap.
- Verification from a streaming recomputation over the final Parquet shards: 12,617 rows, 12,617 unique `instance_id` values, zero rows over context, max shifted input length 130,126, and zero manifest length mismatches.
- Exact replacement/exclusion details are recorded in `context_filter_report.json`.

Tasks excluded by the 2026-05-22 context refresh:

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

If we need the exact paper rows, ask NVIDIA for the training manifest rather than another natural-language clarification.

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
