# TurnPoint

TurnPoint estimates how conversational turns contribute to predictions of conversation
outcomes. It uses a frozen MPNet text encoder and a trainable conversation Transformer
to predict eventual conversation outcomes after each turn. It then estimates turn importance
which measures how much a prediction changes from the preceding turn.

For testing, this guide explains how to reproduce the HFC negotiation analysis.
It produces prediction conversation outcomes, turn importnace scores, and candidate turning points.

For other datasets (Rental, CaSiNo) and combined result tables see [additional analyses](docs/additional-analyses.md).

## Repository overview

```text
turnpoint/
├── run.py             # Main analysis script
├── src/turnpoint/     # TurnPoint package: loading, encoding, modelling, and scoring
├── test/              # Example data-preparation and result-summary scripts
├── data/              # Included source workbooks and generated analysis inputs
├── output/            # Generated models and results (created when you run an analysis)
└── docs/              # Instructions for additional example analyses
```

After installation, the key scripts run in this order:

1. **`test/prepare_mit.py`** prepares the supplied HFC workbooks for this replication example; use your own data-processing script for other datasets.
   Alternatively, **`test/prepare_casino.py`** downloads and prepares the CaSiNo example dataset.
2. **`run.py`** loads prepared data, trains and evaluates the model, and saves turn scores and candidate turning points.
3. **`test/summarise_examples.py`** (optional) combines example results into summary tables, following the [additional-analysis instructions](docs/additional-analyses.md#combine-the-result-tables).

**Called automatically by `run.py`** — these modules in `src/turnpoint/` do not need to be run individually:

| Module | Purpose |
| --- | --- |
| `loader.py` | Reads and validates the prepared conversation and outcome files. |
| `encoder.py` | Converts turn text into numerical representations using the frozen pretrained encoder. |
| `model.py` | Defines the conversation model that predicts outcomes after each turn. |
| `train.py` | Trains the model, selects the best validation epoch, and handles saved models. |
| `importance.py` | Calculates prediction changes, turn importance, and candidate summaries. |

## 1. Clone and install

Choose the setup block for your system.

**macOS / Linux**

```bash
git clone https://github.com/bbevis/turnpoint.git
cd turnpoint
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

**Windows PowerShell**

```powershell
git clone https://github.com/bbevis/turnpoint.git
cd turnpoint
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

The pretrained `sentence-transformers/all-mpnet-base-v2` encoder downloads
automatically on the first analysis run.

For the steps below, choose either your IDE or the terminal. In an IDE, select
this repository's `.venv` Python interpreter. Each script has an **EDIT THESE SETTINGS**
block near the top: edit it, save the file, and click Run with no script arguments.
Paths are resolved from the repository root automatically.

In the terminal, keep `.venv` active and run the commands from the repository root.
Supplying arguments uses the terminal inputs instead of the script's data/output settings.

## 2. Prepare the HFC data

This example uses AI-generated HFC negotiations from the MIT AI negotiation
competition. The four source workbooks are included in
`data/MIT AI negotiation competition/outcomes/`, with dialogue and measured outcomes.
The example combines four source groups, distinguished by their source and the
negotiation instructions given to the bots:

| Source group | Meaning in this example |
| --- | --- |
| `round2` | Second-round competition conversations using participants’ submitted negotiation instructions |
| `unprompted` | Conversations involving a bot without an additional negotiation-strategy prompt |
| `inject_voss_modified` | Conversations involving a bot instructed to question each offer and accept only after the same offer has been repeated five times |
| `negomate_modified` | Conversations involving a bot instructed to plan in detail, considering goals, priorities, alternatives, the other side’s interests, and trade-offs |

The generated data record these labels in a metadata variable named `condition`.
It identifies each conversation’s source group; it is not an outcome or a model
input. The script setting `CONDITIONS` selects which groups to include.

`test/prepare_mit.py` is an independent, dataset-specific preparation script for
replicating this example; it is not part of the TurnPoint algorithm. Its cleaning
rules reflect how these source files were constructed.

**For your own analysis, you need to prepare and validate your own data.** This
script is not a substitute. Create the TurnPoint input files
(`turns.csv` and the relevant outcome CSVs) from your own sources, choosing cleaning
rules appropriate to your data, then use `run.py` with your folder and outcome names.

To replicate HFC, convert the included workbooks as follows.

**In an IDE…**

Open `test/prepare_mit.py`. The settings near the top are already set for HFC:

```python
SOURCE = "data/MIT AI negotiation competition"
DESTINATION = "data/formatted"
EXERCISE = "hfc"
CONDITIONS = None  # Use all four source groups.
```

Change `SOURCE` or `DESTINATION` if needed, then save and click Run.

**In the terminal…**

```bash
python test/prepare_mit.py "data/MIT AI negotiation competition" data/formatted
```

By default, this combines all four source groups into one dataset. The preparation
script removes appended SVI questionnaire responses and tagged private
preparation/strategy blocks from the model inputs. Unparseable or incomplete
dialogues and conversations without identifiable dialogue from both participants
are excluded and logged.

The expected result is **40,638 conversations**. The generated folder contains
`turns.csv`, `conversation_outcomes.csv`, `individual_outcomes.csv`, an exclusion
log, and `preparation_summary.json`.

## 3. Run the HFC analysis

**In an IDE…**

Open `run.py`. In **EDIT THESE SETTINGS** near the top, use:

```python
DATA = "data/formatted"  # Match DESTINATION from preparation.
CONVERSATION_OUTCOMES = ["value_created"]
INDIVIDUAL_OUTCOMES = ["instrumental", "self", "process", "relationship"]
OUTPUT = "output/hfc_mpnet"  # Choose a new or empty folder.
SAMPLE_SIZE = 0  # Use all conversations.
CHECKPOINT = None  # Train a new model.
```

Save and click Run. Training settings, including `EPOCHS`, `SEED`, and
`LEARNING_RATE`, are directly below this block and apply to both ways of running.

**In the terminal…**

```bash
python run.py --data data/formatted --conversation-outcomes value_created --individual-outcomes instrumental self process relationship --output output/hfc_mpnet
```

This predicts joint `value_created` at the dyad level and each participant’s four SVI outcomes at the individual level.
The runner uses:

- An **80/10/10 split by conversation**: 32,510 training, 4,064 validation, and
  4,064 test conversations, using seed 0.
- Ten training epochs, Adam with learning rate 0.001, and the epoch with the
  lowest validation loss restored for evaluation.
- Training-only outcome standardisation and baseline predictions.
- CPU conversation-model training with one PyTorch thread; text encoding uses
  Apple MPS when available, otherwise CPU.
- Final-turn prediction evaluation and turn scoring for **every test conversation**.

Optionally, try 30 conversations first.

**In an IDE…**

In `run.py`, change only these two settings, then save and click Run:

```python
SAMPLE_SIZE = 30
OUTPUT = "output/hfc_check"
```

**In the terminal…**

```bash
python run.py --data data/formatted --conversation-outcomes value_created --individual-outcomes instrumental self process relationship --sample-size 30 --output output/hfc_check
```

For the full analysis, restore `SAMPLE_SIZE = 0` and `OUTPUT = "output/hfc_mpnet"`
in the IDE, or use the full terminal command above.

## 4. Inspect the results

Start with `output/hfc_mpnet/metrics.csv`. Compare the `model` and `training_baseline` rows for each
outcome: lower MAE and RMSE mean smaller prediction errors. Then examine
`importance_summary.csv` and `candidates.csv` for the turn-level analysis.

The output folder contains:

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

## Common issues

- **Returning in a new terminal:** open the cloned `turnpoint` folder and activate
  the environment again: `source .venv/bin/activate` on macOS/Linux, or
  `.\.venv\Scripts\Activate.ps1` in PowerShell. Installation is only needed once.
- **A script or input file cannot be found:** run commands from the repository root,
  the folder containing `README.md` and `run.py`.
- **The output directory is not empty:** change the `--output` path to a new folder,
  such as `output/hfc_repeat`. The runner protects existing results from being mixed.
