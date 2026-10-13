# TurnPoint

TurnPoint is being developed to estimate how conversational turns contribute to
predictions of interaction outcomes. Users provide their conversations and
measured outcomes in a standard format; the package has no fixed outcome names
or dataset-specific assumptions.

**Current status:** data loading, a turn encoder, and a conversation model with masked self-attention
and outcome training are implemented. A frozen pretrained MPNet encoder represents
turn text; the conversation Transformer is fitted to user-supplied outcomes. Consecutive prefix prediction updates provide outcome-specific
turn importance scores.

## 1. Install in a virtual environment

You need Python **3.10 or later** and a local copy of this repository. A virtual
environment keeps this project's packages separate from other Python projects.
The commands below run in a terminal, not inside the Python prompt.

If you have not cloned the repository, replace `REPOSITORY_URL` with its clone
URL (GitHub's **Code → HTTPS**):

```bash
git clone REPOSITORY_URL turnpoint
cd turnpoint
```

If you already cloned it, open that folder and run the following commands from
the folder containing `pyproject.toml`.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

`-e` installs this checkout in editable mode, so code edits are available without
reinstalling. Pandas and PyTorch are installed automatically.
No separate requirements file is needed: dependencies are in `pyproject.toml`.

Check the installation:

```bash
python -c "import sys, turnpoint; print(sys.executable); print(turnpoint.__file__)"
```

The interpreter path should include `.venv`, and the package path should point
to this checkout's `src/turnpoint` folder. The `.venv` folder is ignored by Git;
each user creates their own environment.

### Using VS Code

1. Open the repository folder in VS Code.
2. Install Microsoft's **Python** extension if it is not installed.
3. Open the Command Palette: **Shift + Command + P** on macOS, or
   **Ctrl + Shift + P** on Windows/Linux.
4. Select **Python: Select Interpreter**, then select this project's `.venv`.
5. Open **Terminal → New Terminal** and run the installation check above.

If the environment is not listed, select **Enter interpreter path** and choose
`.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows.

In a new terminal, activate the existing environment again with the activation
command above; you do not need to recreate or reinstall it. To leave it, run
`deactivate`. See the official [Python venv guide](https://docs.python.org/3/library/venv.html)
and [VS Code environment guide](https://code.visualstudio.com/docs/python/environments).

## 2. Format your data

Users convert their own datasets into the following files. Put them together
in one folder, separate from the package source:

```text
formatted_data/
    turns.csv
    conversation_outcomes.csv    # Optional if individual outcomes are supplied
    individual_outcomes.csv      # Optional if conversation outcomes are supplied
```

### turns.csv — one row per turn

| Required column | Meaning |
|---|---|
| `conversation_id` | Identifier linking the turns belonging to one conversation |
| `turn_number` | Positive integer giving the turn's position in that conversation |
| `speaker_id` | Identifier for the participant who produced the turn |
| `text` | Nonblank text of the turn |

```csv
conversation_id,turn_number,speaker_id,text
chat_1,1,customer,Can you explain this?
chat_1,2,advisor,Of course.
```

Turn numbers must be unique within a conversation; gaps are allowed. The loader
sorts them numerically. Speaker IDs must refer to the same participant throughout
that conversation, but need not be globally unique. Existing labels are preserved.
The loader does not require exactly two speakers or silently remove conversations.

Extra turn columns, such as role, word count, or response time, are retained.
Keep outcome variables in the outcome files, not among the turn features.

### conversation_outcomes.csv — one row per conversation

`conversation_id` is required. Every other column is an outcome applying to the
conversation as a whole; in a dyadic conversation, these are dyad-level outcomes.

```csv
conversation_id,resolved,joint_score
chat_1,1,120
```

### individual_outcomes.csv — one row per participant per conversation

`conversation_id` and `speaker_id` are required. Every other column is an outcome
belonging to that participant. IDs must match the corresponding turns.

```csv
conversation_id,speaker_id,satisfaction,learning
chat_1,customer,5,4
chat_1,advisor,6,3
```

### Outcome and file rules

- Supply at least one outcome file, containing at least one outcome column.
- Choose any outcome names. Continuous scores, binary values, and categorical
  labels can be loaded. Encoding and training losses are future model decisions.
- Leave missing outcome cells empty. They become Python `None`, never zero.
- An absent outcome row produces an empty outcome dictionary for that
  conversation or participant. Present columns with blank cells remain present
  with value `None`.
- Outcome rows must have unique keys and refer to conversations/participants
  present in `turns.csv`.
- Use UTF-8 CSVs with the exact filenames and column names above. IDs are read
  as strings, preserving leading zeros; match their spelling across files.
- Include headers. Quote text containing commas, quotation marks, or line breaks
  using standard CSV quoting. Pandas `DataFrame.to_csv(..., index=False)` handles
  this automatically.

## 3. Load and inspect your conversations

With the environment active, run `python` to open Python, then paste:

```python
from turnpoint import load_conversations

# Replace this with your formatted_data folder's absolute path.
conversations = load_conversations("/absolute/path/to/formatted_data")
conversation = next(conversations)

print(conversation["conversation_id"])
print(conversation["turns"][0])
print(conversation["conversation_outcomes"])
print(conversation["individual_outcomes"])
```

With the example CSVs above, the final two lines print:

```python
{'resolved': 1, 'joint_score': 120}
{'customer': {'satisfaction': 5, 'learning': 4},
 'advisor': {'satisfaction': 6, 'learning': 3}}
```

The function yields one dictionary per conversation. Each contains
`conversation_id`, a list of `turns`, a dictionary of `conversation_outcomes`, and
`individual_outcomes` keyed by speaker ID. To process every conversation:

```python
for conversation in load_conversations("/absolute/path/to/formatted_data"):
    print(conversation["conversation_id"], len(conversation["turns"]))
```

For an actor-to-counterpart analysis, select the counterpart's entry in
`individual_outcomes`. The loader does not select participants, combine outcomes,
or estimate behavioural effects. Numeric and categorical types are inferred
per outcome column by pandas. All CSV tables are currently read into memory;
yielding conversations does not make file reading a streaming operation.

## 4. Encode turns

TurnPoint uses the frozen pretrained Sentence Transformer
[`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2),
pinned to revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
The first use downloads the model from Hugging Face; later uses reuse local files.
It is an English text encoder; suitability for other languages needs separate assessment.

```python
from turnpoint import TurnEncoder, load_conversations

conversation = next(load_conversations("/path/to/formatted_data"))
encoder = TurnEncoder()
encoded = encoder(conversation["turns"])
print(encoded["vectors"].shape)  # Number of turns by 770.
```

Each turn is encoded independently, without access to preceding or future turns.
The 768-dimensional text embedding is concatenated with log(1 + word count) and
an indicator that the speaker changed since the preceding turn (first turn = 0).
Turn position and target-speaker information are added by the conversation model.
Extra input columns are retained by the loader but are not automatically model inputs.

Long turns are tokenised without truncation and split into consecutive,
non-overlapping chunks within MPNet's 384-token limit, including special tokens.
Chunk embeddings are L2-normalised, averaged with weights proportional to their
content-token counts, and L2-normalised again. This preserves coverage of the
whole turn, but does not model relationships across chunks inside the text encoder.

Embeddings are cached in `.cache/turnpoint/mpnet.sqlite3`, ignored by Git.
Cache keys include exact text, pretrained revision, and chunking policy.
Only text embeddings are cached; speaker metadata is calculated per conversation.
The cache may contain sensitive information derived from your text; keep it local.
A frozen encoder fits nothing to these data, so its cache can be shared across
training, validation, and test splits without fitting on held-out outcomes.
Use `encoder.prepare(texts)` to compute missing embeddings in batches before training.
Apple MPS accelerates encoding when available; otherwise it uses CPU.
For CPU-only encoding, use `TurnEncoder(encoding_device="cpu")`.

## 5. Conversation model

The conversation model is a randomly initialised turn-level Transformer using
PyTorch encoder layers, with a future-turn attention mask. After each turn it
predicts eventual outcomes using only the current and preceding turns. This is
outcome prediction, not next-token generation.

Defaults are two layers, hidden width 128, four attention heads, feed-forward
width 512, ReLU, dropout 0.1, residual connections and post-sublayer layer norms.
Sinusoidal positions encode turn order. For individual outcomes, shared prediction
heads use target-versus-other speaker indicators. These indicators distinguish
whose outcome is predicted, without learning embeddings for named individuals.
In multi-party conversations, other participants are pooled in this indicator.

```python
from turnpoint import ConversationModel

model = ConversationModel(
    input_dim=encoder.output_dim,
    conversation_outcomes={"value_created": "continuous"},
    individual_outcomes={"relationship": "continuous"},
)
```

Outcome definitions accept `"continuous"`, `"binary"`, or an integer class count.
Outputs are a prediction after every turn, grouped by outcome and participant.
Binary and multiclass outputs are logits until converted to probabilities.

## 6. Train and save a model

```python
from turnpoint import load_conversations, train_model

training = list(load_conversations("/path/to/training_data"))
validation = list(load_conversations("/path/to/validation_data"))
fitted = train_model(
    training, validation,
    conversation_outcomes={"value_created": "continuous"},
    individual_outcomes={"relationship": "continuous"},
    checkpoint="model.pt",
)
```

Supply separate conversations in the two splits. Text embeddings stay frozen;
only the conversation Transformer and outcome heads receive gradient updates.
The training objective compares every prefix prediction with the same observed
final outcome. Loss is averaged over turns and observed weighted targets within
each conversation; labelled conversations have equal weight. Continuous targets
are standardised using training-only means and population standard deviations.
Binary targets use binary cross-entropy; multiclass targets use cross-entropy.
Missing outcomes are skipped. There are no reconstruction or sparsity penalties.

Defaults are ten epochs, Adam learning rate 0.001, gradient clipping at norm 1,
seed 0, and CPU conversation-model training. Validation outcome loss selects the
best epoch after all ten epochs run. Test data must not select epochs.
Optional Python arguments include `epochs`, `learning_rate`, `outcome_weights`,
`encoder_options`, `model_options`, `seed`, and `device`; the command-line runner
keeps these settings internal. For example, `encoder_options={"encoding_device": "cpu"}`.

The returned dictionary contains `encoder`, `model`, `transforms`, `history`,
`best_epoch`, and `best_validation_loss`. Checkpoints store conversation-model
weights, outcome definitions/transforms, encoder configuration, the pinned MPNet
identity/revision, and training history. Frozen pretrained weights are downloaded
separately and reused locally; they are not duplicated in each checkpoint.

```python
import torch
from turnpoint import TurnEncoder, ConversationModel
from turnpoint.encoder import MODEL_NAME, MODEL_REVISION

saved = torch.load("model.pt", map_location="cpu", weights_only=True)
assert saved["text_model"] == MODEL_NAME
assert saved["text_model_revision"] == MODEL_REVISION
encoder = TurnEncoder(**saved["encoder_options"])
model = ConversationModel(encoder.output_dim, **saved["specifications"], **saved["model_options"])
model.load_state_dict(saved["model"])
encoder.eval()
model.eval()
fitted = {"encoder": encoder, "model": model, "transforms": saved["transforms"]}
```

Earlier GRU checkpoints are incompatible with this architecture. Retain them only
as historical results; train a new model for the MPNet workflow.

## 7. Score turns using prediction updates

After training, predict the final outcome after each turn using only the dialogue
observed so far. A turn's signed update is its prediction minus the prediction
at the preceding turn. Its importance is the absolute size of that update divided by the outcome’s
training-set standard deviation (for continuous outcomes).

```python
from turnpoint import score_turns

# fitted is the dictionary returned by train_model in section 6.
conversation = next(load_conversations("/path/to/test_data"))
scores = score_turns(conversation, fitted)
print(scores.head())
scores.to_csv("turn_scores.csv", index=False)
```

| Column | Meaning |
| --- | --- |
| `prediction` | Predicted final outcome after the current turn |
| `previous_prediction` | Prediction after the preceding observed turn |
| `prediction_change` | Current minus previous prediction; signed update |
| `importance` | Absolute change in training-set standard deviation units for continuous outcomes; probability change for classification |
| `importance_original` | Absolute change in original outcome units |
| `standardized_prediction_change` | Signed change divided by the training-set standard deviation for continuous outcomes; signed probability change for classification |
| `importance_units` | `training_standard_deviation` or `probability` |

The first turn has a prediction but blank previous prediction, change, and
importance: there is no preceding prefix to compare. Gaps in recorded turn
numbers do not change this rule; comparison uses the preceding observed turn.

Rows identify `conversation_id`, `turn_number`, `speaker_id`, outcome `scope`,
`outcome`, `participant_id`, and `class_label`. Conversation outcomes have an
empty participant ID. To examine A's turns against B's evaluations, select A as
`speaker_id` and B as `participant_id`.

Continuous predictions and `prediction_change` use original outcome units.
`importance` uses the standard deviation fitted on training outcomes, so 0.1
means a change of one tenth of a training-set standard deviation. It is not
bounded to 0–1. `importance_original` retains the original-unit magnitude.
For a constant training outcome, the training transform uses a scale of 1.
Classification importance remains an absolute probability change. Binary outputs
use the probability of the second training class; multiclass outputs have a row
per class. Positive changes mean a higher predicted value or class probability.
Scores remain separate for each outcome and are not normalised to sum to one.

Scoring requires one model pass, with no altered dialogue or reference vector.
Attention combines preceding dialogue; attention weights are not these scores.
Large updates identify turns that revise the model's expectation, which may
reveal an outcome rather than produce it. Small updates can occur when a turn
was already anticipated. These scores are predictive updates, not causal effects
or direct measurements of participants' changing feelings. Check prefix
prediction quality and score stability across fitted models before interpreting
rankings. Final-turn predictive performance alone does not validate these scores.

## Troubleshooting

| Problem | What to check |
|---|---|
| `No module named turnpoint` | Activate `.venv`, then run `python -m pip install -e .` from the repository root. |
| `FileNotFoundError` | Pass the folder containing `turns.csv`, not the CSV filename. Check filenames and case. |
| `KeyError` naming a column | Use the required headers exactly, without extra spaces. |
| Invalid/duplicate turn number | Use a positive integer once per conversation position. |
| Outcome keys do not match | Match conversation and participant IDs across files, including leading zeros. |
| `StopIteration` from `next(...)` | The turns file has no conversation rows. |

## Our HFC example — separate from the package

`test/prepare_mit.py` converts the supplied AI-generated **HFC or Rental negotiations**
into TurnPoint's standard files. HFC is the default; `--exercise rental` selects Rental.
All MIT-specific processing stays in this script.

The four HFC source workbooks are included under
`data/MIT AI negotiation competition/outcomes/`. No separate data download is
needed after cloning the repository. With `.venv` active, run from the repository root:

```bash
python -m pip install openpyxl==3.1.5
python test/prepare_mit.py "data/MIT AI negotiation competition" data/formatted
```

The script reads `outcomes/*_hfc_wide_outcomes_deid.xlsx`. These source files
already contain both dialogue and measured outcomes, so the separate negotiation
exports and long-form exports are not read again. Table, chair, and rental files
are ignored. The old combined RDS file is not used.

By default it reads all four supplied HFC conditions: `round2`, `unprompted`,
`inject_voss_modified`, and `negomate_modified`. Conditions remain distinct in
turn metadata; including them does not imply a decision to pool them for analysis.
To convert only a selected condition, use, for example:

```bash
python test/prepare_mit.py "data/MIT AI negotiation competition" data/formatted --conditions round2
```

Outputs in `data/formatted/`:

| File | Contents |
| --- | --- |
| `turns.csv` | Negotiation dialogue, participant role, agent ID, condition, source filename and Excel row |
| `conversation_outcomes.csv` | Joint `value_created`, copied from the source |
| `individual_outcomes.csv` | Each participant's `instrumental`, `self`, `process`, and `relationship` SVI scores |
| `excluded_conversations.csv` | Source filename, Excel row, and reason for every excluded conversation |

**SVI questionnaire responses are appended to the source conversations. They are
removed from the input text to avoid exposing outcome answers to the model.**
Tagged preparation and strategy blocks are also removed, preserving any spoken
dialogue following them. A questionnaire response is identified by numbered
items 1 and 16 and agreement-rating language. Unexpected dialogue following a
questionnaire is excluded rather than silently discarded.

Participant IDs are the source roles `consultant` and `coo`. Outcomes are matched
using `role_1`/`role_2`, not inferred from who speaks first. Original agent
identifiers remain metadata. Conversation IDs combine condition, HFC, and the
source Excel row; turn numbers retain the original message positions, with gaps
where questionnaire messages were removed. Missing outcomes stay missing.

The converter excludes conversations with unparseable/truncated lists,
unidentified speakers, or no dialogue from both participants. It reports counts
for each source workbook and records each exclusion. Some source cells reach
Excel's 32,767-character limit; incomplete lists cannot be recovered from those
cells and are excluded. Source workbooks are streamed without modification;
rerunning conversion replaces the four output files.

Load the generated inputs using:

```python
conversations = load_conversations("data/formatted")
```

Other datasets need their own preparation scripts, not changes to `turnpoint`.

**Data included in Git:** only these four source workbooks (41.81 MiB total):

- `round2_hfc_wide_outcomes_deid.xlsx`
- `unprompted_hfc_wide_outcomes_deid.xlsx`
- `inject_voss_modified_hfc_wide_outcomes_deid.xlsx`
- `negomate_modified_hfc_wide_outcomes_deid.xlsx`

`.gitignore` excludes all other files under `data/`, including the generated CSVs,
plus Excel files elsewhere, archives, model checkpoints, and output directories.
The four exact source filenames above are explicitly allowed. Adding another
source workbook requires an intentional change to `.gitignore`.

## 8. Run the model

There is one runner, `run.py`. **Provide your formatted data folder and outcome
names.** It trains one model with a fixed **80% training / 10% validation / 10%
test** conversation split. Model and training settings have internal defaults.

Install dependencies and the package once, from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Then run, for example:

```bash
python run.py --data /path/to/formatted_data --conversation-outcomes score --individual-outcomes satisfaction
```

Use your own outcome column names. Include conversation outcomes, individual
outcomes, or both. List multiple names separated by spaces. Outcomes are
continuous by default; use `name:binary` for a binary outcome or `name:3` for
three categories. No JSON is required for normal use.

For all prepared HFC conversations:

```bash
python run.py --data data/formatted --conversation-outcomes value_created --individual-outcomes instrumental self process relationship --output output/hfc_full_split
```

| Input | What you provide |
| --- | --- |
| `--data` | Folder containing your standard CSV files |
| `--conversation-outcomes` | Conversation outcome column names, if used |
| `--individual-outcomes` | Individual outcome column names, if used |
| `--sample-size` | Optional conversation limit; default `0` uses all conversations |
| `--output` | Optional results folder; default `output/run`, which must be new or empty |

**Automatic settings:**

| Setting | Value |
| --- | --- |
| Sample | All conversations unless a limit is supplied |
| Split | 80% training, 10% validation, 10% test |
| Epochs | 10, then restore the best validation epoch |
| Optimisation | Adam, learning rate 0.001 |
| Turn scoring | Up to 10 held-out test conversations |
| Seed / compute | Seed 0; CPU with one PyTorch thread |

For the 40,638 prepared HFC conversations, this gives **32,510 training, 4,064
validation, and 4,064 test conversations**. Whole conversations stay together.
Sampling and splitting use seed 0. The runner first reserves 20% of conversations,
then divides that reserve equally between validation and testing. Integer
rounding can slightly change proportions for other sample sizes. At least ten
conversations are required.

Outcome scaling and baselines use training
conversations only. Training updates the model weights; validation selects the
best epoch; testing evaluates the selected model. All ten epochs run before the
best epoch is restored. Test conversations never select epochs or fit
preprocessing. The runner prints actual split counts and saves their identities.

Continuous evaluation reports MAE, RMSE, and R-squared against training-mean
predictions. Classification evaluation reports accuracy and log loss against
training class frequencies. Final-turn predictions are evaluated for every test
conversation. Missing labels are omitted from metrics; undefined R-squared stays
blank. Training must contain the required target observations and classes;
held-out labels must belong to the training classes.

All outputs are in the chosen results folder, without separate fold folders:

| Output | Contents |
| --- | --- |
| `configuration.json`, `splits.csv` | Actual parameters, split fractions/counts, environment versions, input checksums, and split membership |
| `model.pt`, `history.csv` | Best weights/preprocessing and per-epoch training/validation losses |
| `training_summary.json` | Selected epoch, validation loss, and training time |
| `predictions.csv`, `metrics.csv` | Test predictions and baseline comparisons |
| `scores.csv` | Prefix predictions, signed updates, and absolute turn importance for every test conversation |

Every test conversation receives turn scores. Individual scores
include own- and counterpart-outcome changes; select different `speaker_id` and
`participant_id` values for counterpart outcomes. Input tables are loaded into
memory even when selecting a smaller sample.

Use `--sample-size 30 --output output/hfc_split_check` with the HFC command for
a small real-data execution check. Keep outputs under `output/` or `outputs/`
so Git ignores them. Use a new folder for each run; nonempty output folders are
rejected. Model and training settings are internal rather than command-line
options. See `python run.py --help` for the user inputs.

Architecture defaults are frozen 768-dimensional MPNet embeddings plus two
metadata values, and a trainable Transformer with two layers, width 128, four
attention heads, and dropout 0.1. Gradient norm is clipped at 1. The exact MPNet
revision is saved in configuration.json and model.pt. Cached embeddings and model
downloads stay local and are not uploaded to GitHub.

Earlier cross-validation checks are historical outputs from the previous runner.
They are retained with their original configurations and are not results from
this single-split protocol.

Pinned dependencies were recorded on macOS with Python 3.13. Seeds and
PyTorch deterministic operations support repeat runs in the same environment;
exact results across platforms, library versions, or devices are not guaranteed.

Outputs from earlier turn-replacement scoring are obsolete and must not be interpreted
as prefix-update scores. Use a new output folder when running the current workflow.

## Full scoring and preliminary summaries

To score an existing MPNet model without retraining, supply its checkpoint and
the original formatted data. The runner verifies input file checksums and uses
only the saved test split. Choose a new output folder.

```bash
python run.py --data data/formatted --checkpoint output/hfc_mpnet/model.pt --output output/hfc_scored
```

Both normal training runs and checkpoint scoring now save:

| File | Contents |
| --- | --- |
| `scores.csv` | Every test turn's predictions, signed updates and absolute importance |
| `importance_summary.csv` | Counts, mean, sample SD, median, quartiles, range, positive/negative/zero updates, and candidate thresholds per outcome/relation/class |
| `candidates.csv` | Candidate turn–outcome flags for shared and counterpart outcomes, without dialogue text |
| `scoring_summary.json` | Overall coverage, unique flagged turns versus turn–outcome flags, and selection rule |
| `scoring_configuration.json` | Checkpoint-only scoring provenance, including checkpoint checksum |

Individual summaries separate own-outcome and counterpart-outcome updates. Primary
candidate exports use counterpart and conversation outcomes. A candidate must have
positive importance at or above the empirical 95th percentile within its outcome,
target-relation, and class group. Ties are retained, so the flagged proportion need
not be exactly 5%; constant zero updates produce no candidates. This is an exploratory
screening rule, not a statistical significance test or proof of a turning point.
First-turn scores are undefined and excluded from descriptive statistics. In multi-party
data there may be multiple counterpart records per focal turn. One turn can be flagged
for multiple outcomes, so unique turn counts and flag counts are reported separately.
The previous ten-conversation exports remain historical partial analyses.

## Additional real-data examples

Dataset-specific preparation remains under `test/`; the package contains no
exercise names, outcome names, or rating mappings.

### MIT Rental

Place the four `*_rental_wide_outcomes_deid.xlsx` workbooks in the same source
outcomes folder as HFC. Rental files remain ignored by Git under the existing
HFC-only source allowlist; they are local inputs, not automatically published.

```bash
python test/prepare_mit.py "data/MIT AI negotiation competition" data/rental_formatted --exercise rental
python run.py --data data/rental_formatted --conversation-outcomes value_created --individual-outcomes instrumental self process relationship --output output/rental_full
```

The supplied Rental sources yield 40,601 conversations and 816,736 turns; 194
conversations are excluded for unidentified speakers. The same dialogue cleaning
rules as HFC remove appended questionnaires and private preparation/strategy tags.
The fixed split is 32,480 training, 4,060 validation and 4,061 test conversations.

### CaSiNo human negotiations

Source: Chawla et al. (2021), [CaSiNo](https://github.com/kushalchawla/CaSiNo),
[paper](https://aclanthology.org/2021.naacl-main.254/), CC BY 4.0.
The converter downloads the original JSON if missing, using pinned revision
`2f6ed4a6a55110152a7699fbaa8150d6036314be` and verifying its SHA-256.

```bash
python test/prepare_casino.py data/casino_raw/casino.json data/casino_formatted
python run.py --data data/casino_formatted --conversation-outcomes joint_points --individual-outcomes points_scored satisfaction opponent_likeness --output output/casino_full
```

All 1,030 conversations are retained, with 11,919 human text turns after removing
2,378 interface events (`Submit-Deal`, `Accept-Deal`, `Reject-Deal`, `Walk-Away`).
Natural-language offers and acceptance remain. Ratings are explicitly mapped from
extremely negative (1), slightly negative (2), undecided (3), slightly positive (4),
to extremely positive (5); mappings are saved in `preparation_summary.json`.
These ordered ratings are treated as continuous outcomes in this example.
`joint_points` is the sum of supplied participant points, not incremental surplus
above a disagreement baseline. Preferences, demographics, deal event metadata and
strategy annotations are not model inputs. The split is 824/103/103 conversations.

The three datasets are trained and evaluated separately. This demonstrates reuse
across datasets; it is not a test of an HFC-trained model transferring to human data.
Raw downloads, formatted data, cached embeddings, and model outputs remain local.

After the example runs finish, `python test/summarise_examples.py` collates
coverage, primary importance statistics, and model/baseline metrics under
`output/proposal_results/`. Missing runs are explicitly marked pending.
