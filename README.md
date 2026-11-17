# When Do Explicit Verifier Verdicts Help LLM Agents? Evidence Quality and Repair in Self-Correction

Code for the paper. Paper: `ARXIV_LINK_TBD`. Data: `release_data.zip` on the
[latest GitHub release](https://github.com/yinruide/explicit-verifier-verdicts/releases/latest)
(see [Data](#data)).

> **Content warning.** The detoxification data (ParaDetox source sentences, the model's rewrites
> of them and the evaluator's reasons, inside the released run records) contain offensive and
> toxic language.

## Abstract

LLM agents in generate-check-revise loops accept or revise their outputs based on reports from imperfect automated verifiers, yet how those reports are shown to the agent is usually an ad hoc choice. We show that this choice changes an agent's accept/rewrite decisions and the final outcome of the loop when the evidence is identical. Holding first attempts and verifier scores fixed, we vary only the presentation (no verifier output, raw scores and thresholds, or explicit verdicts, margins and cross-verifier agreement), cross it with a hidden degradation of the scores, and run the full revision loop in text detoxification and mathematical problem solving. In mathematics, explicit fields lowered false accepts by 10.7 points on intact scores and by 1.3 points once every score had been replaced; stating each verdict accounted for most of the gain without raising false rejects, and lowered terminal failure by 3.7 points. In detoxification, where the agent already used raw scores as well as the best fixed rule, showing verifier output lowered terminal failure from 0.70 to 0.47 and explicit fields changed single decisions by at most 3 points. The design separates presentation from evidence quality and single decisions from final outputs, and applies to other agents and verifiers.

## Licences

* Code (this repository): **Apache-2.0** (`LICENSE`).
* Data (the released run records): **CC BY-NC-SA 4.0**, non-commercial research use only. The data
  archive carries `LICENSE` (the licence text) and `DATA_LICENSE.md`, which lists every upstream
  source (datasets, verifier models, API providers), its terms and links; users of the data must
  also comply with all of those upstream terms.

## Repository layout

| Path | Contents |
|---|---|
| `src/rafr/` | Python package (`rafr`): channel, formatter (representations A1, B, Verdict-only, B + pi-hat, No-Verifier), controller, evaluators, verifiers, calibration, logging |
| `pipelines/` | Experiment pipeline: state construction, thresholds, degradation channel, manipulation check, Experiments 1 and 2, the extensions, and the mathematics pipeline |
| `analysis/` | Analysis behind every table and figure; `analysis/reproduce.py` drives the reproduction |
| `scripts/gpu/` | GPU-side workers of the mathematics stages (verifier scoring, Experiment 2 loops); see [GPU workers](#gpu-workers-for-the-mathematics-stages) |
| `configs/` | Run configurations (models, seeds, channel, experiments) as used for the paper |
| `tests/` | Offline test suite (no API call, no GPU) |

The data (run records and frozen parameter files) are not in git. They live in a separate data
directory that the code finds through the `RAFR_DATA_DIR` environment variable.

## Installation

Python 3.11 or later.

```bash
python -m pip install -e ".[analysis,math,dev]"
```

`requirements.txt` lists the exact version of every third-party package the code imports, as
installed in the environment that ran the tests and regenerated the tables (Python 3.11.5, CPU).
The `gpu` extra of `pyproject.toml` pins the versions the GPU stages ran with (torch 2.8.0,
transformers 5.16.1, ...); it is only needed to rerun the verifiers. The `semantic` extra is only
needed for detoxification's descriptive semantic-preservation report.

## Data

The data archive, `release_data.zip` (about 28 MB; 216 MB unpacked), is attached to the latest
GitHub release. Download it, unpack it into the data directory and check every file against the
sha256 in `MANIFEST.tsv`:

```bash
curl -L -O https://github.com/yinruide/explicit-verifier-verdicts/releases/latest/download/release_data.zip
unzip -q release_data.zip      # creates the data directory release_data/
(cd release_data && tail -n +2 MANIFEST.tsv | awk -F'\t' '{print $3 "  " $1}' | sha256sum -c --quiet) \
  && echo "all files match MANIFEST.tsv"
```

`sha256sum -c` names any file whose sha256 differs and exits non-zero (on macOS, replace
`sha256sum -c --quiet` with `shasum -a 256 -c`). Without `unzip` or `sha256sum` (on Windows, for
example), Python does both steps:

```bash
python -c "import zipfile; zipfile.ZipFile('release_data.zip').extractall('.')"
python -c "import csv, hashlib, pathlib, sys; d = pathlib.Path('release_data'); rows = list(csv.DictReader(open(d / 'MANIFEST.tsv', encoding='utf-8'), delimiter='\t')); bad = [r['path'] for r in rows if hashlib.sha256((d / r['path']).read_bytes()).hexdigest() != r['sha256']]; print(len(rows) - len(bad), 'of', len(rows), 'files match MANIFEST.tsv'); bad and sys.exit('differ: ' + ', '.join(bad))"
```

The data directory then holds:

```
release_data/
  runs/      run records of every experiment the paper reports, and the original analysis outputs
  data/      splits, first-attempt states, labels and verifier scores, gate diagnostics, cohorts
  frozen/    parameters fixed before the experiments that use them (see below)
  LICENSE, DATA_LICENSE.md        CC BY-NC-SA 4.0 and the upstream terms
  MANIFEST.tsv, MANIFEST.json     every file with its size and sha256 (the manifests list the
                                  other 288 files, not themselves)
```

Point the code at it:

```bash
export RAFR_DATA_DIR=$PWD/release_data
```

**`frozen/`** sits in the data directory, not in git. It holds the parameters each later stage
reads and that were fixed before that stage ran: the detoxification thresholds (`tau.json`), the
degradation channel (`channel.json`; `channel.candidate.json` is the identical candidate it was
promoted from), the record of the manipulation check (`gate_attempts.json`), and the pi-hat
calibration maps (`calibration_detox.json`, `calibration_math.json`;
`calibration_detox_v1_no_prior_correction.json` is the superseded fit before the prior correction).
The mathematics thresholds and channel are part of its run (`runs/math_formal_n9000/fit.json`,
`gate_1.json`). The files are byte-for-byte the originals: configurations and run records refer to
them by sha256, and they keep the provenance written at the time (paths, commits).

For every step the records hold the controller's decisions and rewrite texts (`decisions.jsonl`,
trajectories), the verifier scores of every scored text (`scores.jsonl`, `text_scores.jsonl`, and
the first-attempt score tables in `data/scores/` and `runs/math_formal_n9000/scores.jsonl`), the
pi-hat values shown to the controller (`pihat.jsonl`), and the evaluator's judgments (first-attempt
labels, `evaluations.jsonl`, `round_judgments.jsonl`, `extension_e_evaluations.jsonl`,
`math_judgments.jsonl`, and the second judge's `panel_grok.jsonl`).

The source datasets are not redistributed as such; they are available on Hugging Face at the
revisions used:

| Dataset | Hugging Face ID | Revision |
|---|---|---|
| ParaDetox | `s-nlp/paradetox` | `958b799a4c3a7c16463632465aaaa8ad083235a8` |
| DeepMath-103K | `zwhe99/DeepMath-103K` | `5cf055d1fe3d7a2eb19719ac020211469736ae44` |

## Reproducing the tables and figures (no API, no GPU)

```bash
make tables  DATA=$RAFR_DATA_DIR    # every table of the paper  -> out/paper/tables.md (+ tables.json)
make figures DATA=$RAFR_DATA_DIR    # both operating-point figures -> out/figures/
make verify  DATA=$RAFR_DATA_DIR    # every regenerated number vs. the original analysis outputs
```

Without `make`: `python analysis/reproduce.py tables --data /path/to/data` (likewise `figures`,
`verify`). Nothing is written into the data directory; everything goes to `out/`.

`make tables` recomputes every analysis from the run records (decisions, trajectories, judgments):

| Paper | Regenerated by | Output under `out/` |
|---|---|---|
| Table 1 (primary), detox/math primary appendix tables, H2 strata, detection rates on D_exp | `python -m analysis.prereg_tables` (detox), `pipelines/math_tables.py` (math) | `detox_primary/`, `math_primary/` |
| Operating points, contrasts, rule comparisons, verdict-following gaps, Experiment 2 extension tables | `analysis/ext_analysis.py`, `analysis/ext_decision_quality.py` | `ext_analysis_*`, `ext_decision_quality_*` |
| Correction ceiling | `analysis/ext_ceiling_summary.py` | `ext_ceiling_summary_math_*` |
| Cue diagnostics, minimum detectable effects | `analysis/paper_appendix_ab.py` | `paper_appendix_ab/` |
| All tables in the paper's layout | `analysis/paper_tables.py` | `paper/tables.md`, `paper/tables.json` |
| Operating-point figures | `analysis/paper_operating_points.py` | `figures/*_operating_points.{pdf,png}` |

Notes:

* The mathematics primary tables are laid out from the run's `report.json`, which
  `MathPipeline.report` computed from the same run records (`python pipelines/math_run.py report`).
  Recomputing that report checks the run's recorded code and path identity, so it only runs at the
  run's original location; `make tables` therefore lays out the stored report.
* The detection-rate tables on D_gate and the calibration tables read the stored gate reports
  (`data/diagnostics/step09_report.json`, `runs/math_formal_n9000/gate_1.json`) and calibration
  fits and diagnostics (`frozen/calibration_*.json`, `runs/*calibration*/diagnostics.json`).
* Two tables are computed in `analysis/paper_tables.py` itself and are marked DERIVED in
  `tables.md`: the final failure rate at each domain's first-attempt failure rate (arithmetic on the
  regenerated rates), and the decomposition of the paired Experiment 2 contrast by primary
  assignment (failure counts from the Experiment 2 trajectories).
* The design-overview figure (Figure 1) is a drawing, not generated by code.

## Rerunning from scratch

Rerunning calls paid APIs and needs a CUDA GPU. Copy `.env.example` to `.env` and fill in the keys.
`make rerun-plan` prints the stages below.

| Stage | APIs | GPU | Recorded cost (list prices) |
|---|---|---|---|
| Detox: E prefilter of the ParaDetox sources, G first attempts, E labels | Anthropic, OpenAI | – | about $80 |
| Detox: verifier scores | – | 1x A100 80GB | about 0.2 GPU-hours |
| Detox: Experiments 1 and 2 | OpenAI, Anthropic, xAI | 1x A100 80GB (Exp. 2 loop) | about $12 API |
| Detox: extensions | OpenAI, Anthropic, xAI | 1x A100 80GB | about $42 API, 2.3 GPU-hours active |
| Math: G first attempts on 9,000 problems | OpenAI (E is Math-Verify, local) | – | about $106 |
| Math: verifier scores | – | 1x A100 80GB | about 39 GPU-hours (recorded session) |
| Math: Experiments 1 and 2 | OpenAI | 1x A100 80GB | about $33 API, about 7 GPU-hours |
| Math: extensions and correction ceiling | OpenAI | 1x A100 80GB | about $101 API, about 17 GPU-hours |

All GPU work used one NVIDIA A100-SXM4-80GB with `torch 2.8.0+cu128` (CUDA 12.8.1, Ubuntu 24.04).
Each entry point documents its arguments in its module docstring.

**Detoxification**, in order (outputs land in `$RAFR_DATA_DIR`, or the repo when it is unset):

1. `python pipelines/step00_prepare_data.py` — ParaDetox sources, E prefilter (Anthropic).
2. `python pipelines/step04_build_states.py` — one first attempt per source (OpenAI).
3. `python pipelines/step05_label_and_split.py judge`, then `... split` — E labels, splits.
4. Verifier scores: `python -m rafr.verifiers.probe_run score` (D_fit and pool) and
   `python pipelines/step06_score_states.py reuse | inputs | score | install | verify` (D_gate,
   D_exp on the GPU; `reuse` imports the D_fit and pool scores).
5. `python pipelines/step07_fit_tau.py`, `python pipelines/step08_tune_channel.py`,
   `python pipelines/step09_gate.py` — thresholds, channel, manipulation check (local).
6. `python pipelines/step11_exp1.py --confirm` — Experiment 1 (OpenAI).
7. `python pipelines/step12_exp2.py x1`, `loop` (GPU, OpenAI), `judge`, `panel`, `report` —
   Experiment 2.
8. Extensions: `pipelines/ext_calibrate.py`, `pipelines/ext_cohort.py`,
   `pipelines/ext_exp1.py --cohort {e_incorrect,e_correct}`,
   `pipelines/ext_exp2.py starts | loop | judge | panel`.

**Mathematics**: `pipelines/math_formal_sample.py` (problem sample from DeepMath-103K), then
`pipelines/math_run.py` (`prepare`, `generate`, `split`, `fit`, `channel-check`, `gate`, `exp1`,
`judge`, `report`, with `--config configs/math/formal.json`) on a CPU machine, with the GPU stages in
`scripts/gpu/` (below), and `pipelines/math_ext_run.py` for the extensions and the correction
ceiling.

### GPU workers for the mathematics stages

`scripts/gpu/` holds the GPU-side workers that ran the mathematics verifiers:

| Script | Stage |
|---|---|
| `math_step06_remote.py` | verifier scoring of the first attempts (`fetch`, `check203`, `loopfit`, `score`, ...) |
| `math_exp2_remote.py` | the primary Experiment 2 loop (V1, V2 on the GPU, OpenAI controller) |
| `math_ext_remote.py` | the extension's Experiment 2 loops |
| `math_vprobe2_remote.py`, `math_vprobe_remote.py` | helpers the scoring worker imports (model download, prechecks) |

These scripts were written for RunPod pods and are released as they ran, with only their path
setup adjusted to this directory; **they have not been re-executed in this repository.** The
laptop-side drivers that started the pods, pushed the inputs and pulled the results back over SSH
are not released.

* **GPU and software**: one NVIDIA A100-SXM4-80GB; the runs used the image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8.1, Ubuntu 24.04, Python 3.12.3,
  torch 2.8.0+cu128) with the `gpu` extra (transformers 5.16.1). Room for both verifiers' weights
  (about 35 GB).
* **Before use**:
  1. Prepare the `--base` directory the workers read and write (`data/` for inputs, `models/` for
     weights, `out/` for results; layouts in each script's docstring). The inputs come from the CPU
     side: `pipelines/math_step06.py` (scoring inputs, thresholds, Experiment 2 first states) and
     `pipelines/math_ext_run.py` (the extension's input bundles).
  2. Download the pinned weights with the `fetch` command (pinned revisions; no Hugging Face token
     is needed for these two models).
  3. Set `OPENAI_API_KEY` in the environment for the Experiment 2 loops (the scripts' messages refer
     to the RunPod UI, where it was set).
  4. The workers refuse to score until the 203-state check (`check203`) reproduces recorded scores
     of 203 D_fit states on the machine; a different GPU or software stack can fail it. Its
     reference rows (`data/check203.json`, `data/reference/<id>.jsonl` under `--base`) come from the
     verifier-selection probe that preceded the study and are not part of the data archive.
  5. Copy the outputs back to the CPU side yourself; `pipelines/math_ext_run.py` and
     `pipelines/math_run.py` then judge and report them.

## Models

| Role | Model | Revision / version |
|---|---|---|
| Generator G and controller M | GPT-5.6 Terra, `gpt-5.6-terra` (OpenAI API), `reasoning_effort` medium; output cap 8,192 (detox), 16,384 (math) | API model id |
| Evaluator E (detox) | Claude Opus 5, `claude-opus-5` (Anthropic API), output cap 4,096 | API model id |
| Second judge (detox) | Grok 4.6, `grok-4.6` (xAI API), temperature 0.0, output cap 4,096 | API model id |
| Evaluator E (math) | Math-Verify 0.9.0 | `math-verify==0.9.0` |
| V1 (detox) | `s-nlp/roberta_toxicity_classifier` | `048c25bb1e199b98802784f96325f4840f22145d` |
| V2 (detox) | `google/shieldgemma-9b` | `b8b636016df4540721a098c7aab91c97ec6ee508` |
| V1 (math) | `zai-org/GLM-Z1-9B-0414` | `b221b06fefb23ca320922cf6e68ab5f2fb82de81` |
| V2 (math) | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `6a6f4aa4197940add57724a7707d069478df56b1` |

ShieldGemma is gated on Hugging Face: accept the Gemma Terms of Use on the model page with your
account before downloading, and set `HF_TOKEN`.

## Third-party terms

Datasets, models and API outputs have their own terms; `DATA_LICENSE.md` in the data archive lists
them with links. In short: ParaDetox (Hugging Face card: OpenRAIL++; GitHub: CC0-1.0),
DeepMath-103K (MIT), `s-nlp/roberta_toxicity_classifier` (OpenRAIL++; card metadata also lists
CC BY-NC-SA), ShieldGemma (Gemma Terms of Use), GLM-Z1-9B-0414 (MIT), DeepSeek-R1-Distill-Llama-8B
(MIT, derived from Llama 3.1), Math-Verify (Apache-2.0), and the OpenAI, Anthropic and xAI terms for
the model outputs in the run records.

## Citation

```bibtex
@misc{TBD,
  title  = {When Do Explicit Verifier Verdicts Help LLM Agents? Evidence Quality and Repair in Self-Correction},
  author = {TBD},
  year   = {TBD},
  note   = {arXiv:TBD}
}
```
