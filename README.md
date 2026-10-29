# TurnPoint

TurnPoint estimates how conversational turns contribute to predictions of negotiation
outcomes. It uses a frozen MPNet text encoder and a trainable conversation Transformer
to predict eventual outcomes after each turn. Turn importance measures how much a
prediction changes from the preceding turn. These are predictive updates, not causal
effects or direct measurements of participants’ feelings.

This guide reproduces the HFC negotiation analysis using the source data included in
this repository. Additional instructions below cover the Rental and CaSiNo analyses,
which train separate models on separate datasets.

## 1. Clone and install

The recorded replication environment is **Python 3.13 on macOS ARM**. Use Python
3.13 to follow the pinned environment as closely as possible. Run these commands in
a terminal:

```bash
git clone https://github.com/bbevis/turnpoint.git
cd turnpoint
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows PowerShell, replace the environment creation and activation commands with:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Run all remaining commands from the repository root with the environment active.
The first analysis run needs internet access to download the pretrained
`sentence-transformers/all-mpnet-base-v2` encoder. Model downloads and cached
embeddings are reused on later runs and remain local.

## 2. Prepare the HFC data

The four HFC source workbooks are included in
`data/MIT AI negotiation competition/outcomes/`; no separate data download is needed.
Convert them to the CSV inputs used by TurnPoint:

```bash
python test/prepare_mit.py "data/MIT AI negotiation competition" data/formatted
```

This uses all four conditions: `round2`, `unprompted`, `inject_voss_modified`, and
`negomate_modified`. It removes appended SVI questionnaire responses and tagged
private preparation/strategy blocks from the model inputs. Unparseable or incomplete
dialogues and conversations without identifiable dialogue from both participants
are excluded and logged.

The expected result is **40,638 conversations**. The generated folder contains
`turns.csv`, `conversation_outcomes.csv`, `individual_outcomes.csv`, an exclusion
log, and `preparation_summary.json`. Preparation prints counts for each workbook;
rerunning it replaces the generated files.

## 3. Run the HFC analysis

```bash
python run.py --data data/formatted --conversation-outcomes value_created --individual-outcomes instrumental self process relationship --output output/hfc_mpnet
```

This predicts joint `value_created` and each participant’s four SVI outcomes.
The runner uses:

- An **80/10/10 split by conversation**: 32,510 training, 4,064 validation, and
  4,064 test conversations, using seed 0.
- Ten training epochs, Adam with learning rate 0.001, and the epoch with the
  lowest validation loss restored for evaluation.
- Training-only outcome standardisation and baseline predictions.
- CPU conversation-model training with one PyTorch thread; text encoding uses
  Apple MPS when available, otherwise CPU.
- Final-turn prediction evaluation and turn scoring for **every test conversation**.

For a small execution check first, append
`--sample-size 30` and replace the output path with `output/hfc_check`. This checks
that the workflow runs; it does not reproduce the full analysis.

Always choose a new or empty output directory. Generated data, checkpoints, cached
embeddings, and results are ignored by Git. Seeds and deterministic PyTorch operations
support repeat runs in the same environment, but exact numerical agreement across
platforms, library versions, or devices is not guaranteed.

## 4. Inspect the results

Results are saved under `output/hfc_mpnet/`:

| File | Purpose |
| --- | --- |
| `metrics.csv` | Test MAE, RMSE, and R-squared for the model and training-mean baseline |
| `predictions.csv` | Final-turn predictions and observed outcomes |
| `importance_summary.csv` | Turn-importance statistics by outcome and own/counterpart relation |
| `candidates.csv` | Candidate turn–outcome flags for conversation and counterpart outcomes |
| `scores.csv` | Detailed prefix predictions, signed changes, and absolute importance |
| `configuration.json`, `splits.csv` | Settings, environment versions, input checksums, and split membership |
| `model.pt`, `history.csv`, `training_summary.json` | Selected model, epoch losses, and training summary |
| `scoring_summary.json` | Scoring coverage and candidate counts |

For continuous outcomes, importance is the absolute prediction change divided by
that outcome’s training-set standard deviation. The first turn has no preceding
prediction, so its change and importance are undefined. Individual scores distinguish
changes in the speaker’s own outcome from changes in the counterpart’s outcome.

Candidates have positive importance at or above the empirical 95th percentile
within each outcome, target-relation, and class group. Ties are retained. This is an
exploratory screening rule, not a significance test or proof of a turning point.
Inspect prediction performance against the baseline before interpreting rankings;
final-turn performance alone does not validate turn scores.

## 5. Additional datasets and combined summary

These analyses reuse the same workflow but train separate models. They do not test
whether an HFC-trained model transfers to another dataset.

### MIT Rental

Rental source workbooks are **not included in the repository**. To reproduce this
analysis, obtain the four `*_rental_wide_outcomes_deid.xlsx` workbooks and place them
in `data/MIT AI negotiation competition/outcomes/`. Without these inputs, skip Rental;
cloning the repository is sufficient for HFC, but not Rental.

```bash
python test/prepare_mit.py "data/MIT AI negotiation competition" data/rental_formatted --exercise rental
python run.py --data data/rental_formatted --conversation-outcomes value_created --individual-outcomes instrumental self process relationship --output output/rental_full
```

The sources used in the original analysis yielded **40,601 conversations** and
816,736 turns, with 194 conversations excluded for unidentified speakers.

### CaSiNo human negotiations

The preparation script downloads the [CaSiNo dataset](https://github.com/kushalchawla/CaSiNo)
(Chawla et al., 2021; CC BY 4.0) if it is missing. The source revision is pinned
and its SHA-256 checksum is verified.

```bash
python test/prepare_casino.py data/casino_raw/casino.json data/casino_formatted
python run.py --data data/casino_formatted --conversation-outcomes joint_points --individual-outcomes points_scored satisfaction opponent_likeness --output output/casino_full
```

Expected coverage is **1,030 conversations** and 11,919 text turns after removing
interface events. Satisfaction and liking are mapped to ordered 1–5 ratings and
treated as continuous outcomes. Joint points are the sum of supplied participant
points, rather than surplus above a disagreement baseline.

### Combine the result tables

The summary script expects HFC scores in `output/hfc_scored/`. Create that folder
by scoring the saved HFC model, without retraining:

```bash
python run.py --data data/formatted --checkpoint output/hfc_mpnet/model.pt --output output/hfc_scored
python test/summarise_examples.py
```

Checkpoint scoring verifies the formatted input checksums and uses the saved test
split. The summary script combines HFC, Rental, and CaSiNo results into
`output/proposal_results/dataset_summary.csv`, `importance_summary.csv`, and
`prediction_metrics.csv`. Missing dataset runs are marked `pending`; the summary
can therefore also be generated after completing HFC alone.
