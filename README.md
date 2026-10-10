# TurnPoint

TurnPoint is being developed to estimate how conversational turns contribute to
predictions of interaction outcomes. Users provide their conversations and
measured outcomes in a standard format; the package has no fixed outcome names
or dataset-specific assumptions.

**Current status:** data loading, a turn encoder, and a conversation model with masked self-attention
and joint training are implemented. Neural components start untrained and can
now be fitted to user-supplied outcomes. Turn masking and outcome-specific
importance scores are implemented.

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

`src/turnpoint/encoder.py` uses PyTorch embeddings, a GRU, and a sparse
autoencoder. It contains no dataset-specific code or named outcome variables.

Build the vocabulary from **training conversations only**, after deciding your
conversation-level split. Reuse that vocabulary for validation and testing;
unseen tokens map to an unknown token. For a small runnable demonstration:

```python
from turnpoint import TurnEncoder, build_vocabulary

training_turns = [
    {"speaker_id": "customer", "text": "Can you explain this?"},
    {"speaker_id": "advisor", "text": "Of course."},
]
vocabulary = build_vocabulary((turn["text"] for turn in training_turns), min_count=1)
encoder = TurnEncoder(vocabulary)
result = encoder(training_turns)
print(result["vectors"].shape)  # torch.Size([2, 514])
```

In actual use, pass `conversation["turns"]` from the loader. Lowercase word and
punctuation tokens receive learned embeddings. Defaults are 100 embedding
entries, 128 GRU units, and 512 sparse ReLU features. Change them with
`TurnEncoder(vocabulary, embedding_dim=100, hidden_dim=128, sparse_dim=512)`.
`build_vocabulary` defaults to at most 20,000 words/punctuation tokens appearing
at least twice. Its tokenisation is a simple baseline, not a language-specific
segmenter; languages without whitespace-separated words may need a different
text preparation choice.

The return dictionary contains:

| Key | Contents |
|---|---|
| `vectors` | One row per turn: sparse features, log(1 + word count), speaker-change indicator |
| `semantic_features` | The sparse features alone, without metadata |
| `reconstruction_loss` | Mean squared error reconstructing the dense GRU vectors |
| `sparsity_loss` | Mean absolute sparse activation, to be weighted by the trainer |

The first speaker-change indicator is zero. Later values are one when the
speaker differs from the previous turn, otherwise zero. This keeps arbitrary
speaker names out of the numerical features. For dyads, the change sequence
captures the speaker pattern up to relabelling; for multi-party conversations
it does not fully represent identity. Original IDs remain in the loaded data
for assigning individual outcomes. Pass one complete, ordered conversation
per encoder call; metadata must not span separate conversations.

**These vectors are initially untrained.** The training function below combines
outcome prediction with reconstruction and sparsity losses. Reconstruction alone is not the training objective. No
outcome information enters the encoder directly. Unit-length decoder columns
prevent lowering the sparsity penalty merely by rescaling decoder weights.

The model uses PyTorch's existing [GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
and embedding/linear layers. No separate neural-network implementation is needed.

## 5. Predict outcomes after each turn

`src/turnpoint/model.py` defines `ConversationModel`, using PyTorch's
[TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html).
It is a turn-level Transformer with masked self-attention, built from PyTorch
encoder layers. Each layer contains multi-head self-attention, a ReLU
feed-forward network, residual connections, and layer normalisation after each
residual addition. There is no encoder-decoder cross-attention or pretrained
checkpoint. It takes turn vectors and original speaker IDs, without outcome
labels. A future-turn mask restricts each prediction to the current and earlier
turns, including the turn being evaluated.
Sinusoidal positions encode order. Defaults are two layers, four attention heads,
128 hidden dimensions, and 0.1 dropout; all are configurable.

Specify the names and types of the outcomes you want to model:

| Type specification | Output after each turn |
|---|---|
| `"continuous"` | One regression score |
| `"binary"` | One logit; use sigmoid to obtain a probability |
| An integer >= 2, e.g. `3` | That many class logits; use softmax for probabilities |

For example, using `encoder` and `training_turns` from the previous section:

```python
import torch
from turnpoint import ConversationModel

model = ConversationModel(
    input_dim=514,  # Default encoder: 512 sparse features + 2 metadata entries.
    conversation_outcomes={"resolved": "binary", "joint_score": "continuous"},
    individual_outcomes={"satisfaction": "continuous", "learning": "continuous"},
)
encoder.eval()
model.eval()
with torch.no_grad():
    encoded = encoder(training_turns)
    predictions = model(encoded["vectors"], [t["speaker_id"] for t in training_turns])

print(predictions["conversation_outcomes"]["joint_score"].shape)  # [2, 1]
print(predictions["individual_outcomes"]["customer"]["satisfaction"].shape)  # [2, 1]
probability_resolved = predictions["conversation_outcomes"]["resolved"].sigmoid()
```

Each row predicts the same eventual outcome after another turn has been observed.
Outputs are tensors, so gradients can flow through the model into the encoder
during joint training. If you change the encoder's `sparse_dim`, set `input_dim`
to that dimension plus two. Hidden dimensions must be divisible by attention heads.

For individual outcomes, the model conditions a shared prediction layer on which
turns were spoken by the participant being evaluated. Thus a customer's evaluation
and an advisor's evaluation receive distinct predictions without assigning numeric
meaning to their names. Renaming arbitrary participant IDs simply renames the
corresponding output keys. Conversation-level predictions use a separate shared
stream. In multi-party data, target indicators group all non-target speakers
together; they do not fully distinguish every other participant's identity.

**The outputs are not meaningful predictions until training.** The training
function below fits both neural components against observed outcomes, skipping
missing targets and combining outcome, reconstruction, and sparsity losses.
This model predicts outcomes; use `score_turns` below to compute turn importance.

## 6. Train on your outcomes

`src/turnpoint/train.py` provides `train_model`. Prepare separate training and
validation folders in the standard format, then select your outcome names and
types. Split **whole conversations**, not turns from the same conversation.
Decisions about shared participants, agents, or other dependencies belong to your
dataset preparation script. Keep a separate test split untouched for final
assessment; this function does not evaluate it.

```python
from turnpoint import load_conversations, train_model

fitted = train_model(
    training=load_conversations("/path/to/training_data"),
    validation=load_conversations("/path/to/validation_data"),
    conversation_outcomes={"resolved": "binary", "joint_score": "continuous"},
    individual_outcomes={"satisfaction": "continuous", "learning": "continuous"},
    epochs=10,
    checkpoint="/path/to/existing/output_folder/best_model.pt",
)
print(fitted["best_epoch"])
print(fitted["best_validation_loss"])
```

Replace all paths and outcome names with your own. You can supply conversation
outcomes, individual outcomes, or both. Each selected outcome needs observed
training values. For an outcome with three classes, specify `"outcome_name": 3`;
all three labels must be represented in training. Validation labels must belong
to the training categories. Binary numeric 0/1 targets preserve their meaning;
other binary labels must have two training categories. Their ordering is stored
in `fitted["transforms"]`.

The function:

- Builds the vocabulary from training text only.
- Standardises each continuous outcome using its training mean and population
  standard deviation (scale 1 for a constant target). Individual-outcome scales
  are pooled across participants, not fitted separately for arbitrary IDs.
- Learns binary/category label mappings from training targets only.
- Predicts each final outcome after every observed turn, supervising each prefix
  with the same final target.
- Skips missing targets and conversations with no positively weighted observed
  target. Each split must contain at least one labelled conversation.
- Trains both components jointly using Adam, with gradient norm clipping at 1.
- Restores the encoder and model from the epoch with the lowest validation
  **outcome loss**, excluding reconstruction and sparsity penalties.

Training uses one conversation per optimiser step. Each target loss averages
across turns, then observed target losses are averaged within each conversation.
Each observed participant/outcome pair is a target; missing targets are omitted
from that average. Epoch losses average equally across labelled conversations.
Longer conversations therefore receive no extra weight merely for having more
turns. Both splits are materialised in memory; this is an initial single-
conversation training implementation, not a batched training system.

### Training settings

| Argument | Default | Meaning |
|---|---|---|
| `epochs` | 10 | Number of training passes |
| `learning_rate` | 0.001 | Adam learning rate |
| `reconstruction_weight` | 0.1 | Weight of sparse-autoencoder reconstruction loss |
| `sparsity_weight` | 0.001 | Weight of mean absolute sparse activation |
| `seed` | 0 | PyTorch random seed |
| `device` | `"cpu"` | PyTorch device for both neural components |
| `checkpoint` | `None` | Optional file to save the best weights and preprocessing |

Pass constructor settings through `encoder_options`, `model_options`, and
`vocabulary_options`. For example, `encoder_options={"sparse_dim": 256}` or
`model_options={"hidden_dim": 64, "num_heads": 4}`. The training function connects
the encoder's output width to the model automatically.

All outcomes default to weight 1. To change their relative contributions, pass
`outcome_weights={"conversation_outcomes": {"joint_score": 2.0}}`. The weighted
average is normalised by the sum of weights of observed targets in each
conversation. Unspecified weights remain 1; weights must be nonnegative.

### Returned models and saved checkpoints

The returned dictionary contains `encoder`, `model`, `vocabulary`, `transforms`,
`history`, `best_epoch`, and `best_validation_loss`. History records training
and validation outcome losses for each epoch. The returned neural components
are already in evaluation mode.

Use the trained components as in section 5, replacing `encoder` and `model` with
`fitted["encoder"]` and `fitted["model"]`. Continuous outputs are in standardised
units. Recover original units with `prediction * scale + mean`, using
`fitted["transforms"][scope][outcome_name]`, where scope is
`"conversation_outcomes"` or `"individual_outcomes"`. Binary/multiclass outputs
remain logits; sigmoid/softmax gives probabilities. Class probabilities follow
the order in the stored `classes` list.

When `checkpoint` is supplied, its parent directory must already exist. The
file stores both state dictionaries, vocabulary, outcome transforms and types,
constructor options, best epoch, validation loss, and history. For example:

```python
import torch
from turnpoint import TurnEncoder, ConversationModel

saved = torch.load("/path/to/best_model.pt", map_location="cpu", weights_only=True)
encoder = TurnEncoder(saved["vocabulary"], **saved["encoder_options"])
model = ConversationModel(
    encoder.sparse_encoder.out_features + 2,
    **saved["specifications"], **saved["model_options"],
)
encoder.load_state_dict(saved["encoder"])
model.load_state_dict(saved["model"])
encoder.eval()
model.eval()
```

Checkpoints support prediction; they do not contain optimiser state for resuming
an interrupted run. Validation loss is a model-selection measure, not evidence
of predictive quality by itself. Assess predictions on held-out conversations
and against simpler baselines before interpreting turn importance.

## 7. Score each turn by masking its content

After training, construct a reference mask from **training conversations only**,
using the fitted encoder. The mask is the mean semantic representation across
training turns. Reuse it when scoring validation or held-out test conversations.

```python
from turnpoint import fit_mask, score_turns

# fitted is the dictionary returned by train_model in section 6.
# Reload training conversations to construct the reference mask.
mask = fit_mask(load_conversations("/path/to/training_data"), fitted["encoder"])
conversation = next(load_conversations("/path/to/test_data"))
importance = score_turns(conversation, fitted, mask)

print(importance["scores"].head())
importance["scores"].to_csv("turn_scores.csv", index=False)
importance["trajectories"].to_csv("turn_trajectories.csv", index=False)
```

Each turn is masked separately by replacing its semantic features with the
reference vector. Its word count, speaker information, and sequence position
remain. Every later turn retains its observed content. No outcome labels are
needed for scoring.

The difference is **original prediction minus masked prediction**:

| Score | Meaning |
| --- | --- |
| `immediate` | Signed difference directly after the masked turn |
| `dynamic` | Mean absolute difference from that turn through the conversation end |
| `final` | Signed difference at the end of the conversation |

`dynamic` is the primary magnitude score. Averaging prevents earlier turns from
scoring higher merely because more turns follow them. Scores remain separate
for each outcome and participant; they are not normalised to sum to one.

Both tables identify the conversation, focal `turn_number`, focal `speaker_id`,
outcome `scope`, `outcome`, and outcome owner `participant_id`. A conversation
outcome has an empty participant ID. Thus, filtering for one focal speaker and
another participant selects that speaker's turns scored against the other's
outcomes. `trajectories` also contains `prediction_turn_number`, `original`,
`masked`, and `difference`, showing how the change develops at each later turn.

Continuous differences use the original outcome units. Binary differences use
probabilities of the second training class; `class_label` identifies it.
Multiclass outcomes have one row per class, using probability differences.
A positive signed score means the original content increased that prediction
relative to the reference mask.

Save the mask alongside your model with `torch.save(mask, "mask.pt")` and load
with `torch.load("mask.pt", weights_only=True)`. Recompute it after refitting
model weights. The mask is not automatically included in the training checkpoint.

The training mean provides a reference, but is not guaranteed to be neutral or
represent a naturally occurring turn. Check whether rankings survive alternative
training-derived masks and refitted models. Interpret scores only after assessing
held-out predictive performance. These are changes in model predictions, not
estimates of what participants would actually have done with different dialogue.
Scoring requires one original pass plus one pass per turn; complete trajectories
produce quadratic numbers of rows, so start with short conversations.

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

`test/prepare_mit.py` converts the supplied AI-generated **HFC negotiations only**
into TurnPoint's standard files. All MIT-specific processing stays in this script.

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

Vocabulary, outcome scaling, baselines, and the reference mask use training
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
| `mask.pt`, `scores.csv`, `trajectories.csv` | Training-derived mask and turn importance for up to ten test conversations |

The first up to ten test conversations in deterministic split order receive turn
scores; they are not a representative sample by design. Individual scores
include own- and counterpart-outcome changes; select different `speaker_id` and
`participant_id` values for counterpart outcomes. Input tables are loaded into
memory even when selecting a smaller sample.

Use `--sample-size 30 --output output/hfc_split_check` with the HFC command for
a small real-data execution check. Keep outputs under `output/` or `outputs/`
so Git ignores them. Use a new folder for each run; nonempty output folders are
rejected. Model and training settings are internal rather than command-line
options. See `python run.py --help` for the five user inputs.

Architecture defaults remain 100-dimensional embeddings, 128 GRU units, 512
sparse features, and a Transformer with 2 layers, width 128, 4 attention heads,
and dropout 0.1. Vocabulary maximum is 20,000 tokens with minimum count 2.
Gradient norm is clipped at 1. Reconstruction and sparsity weights are 0.1 and
0.001. These are starting settings, not tuned settings.

Earlier cross-validation checks are historical outputs from the previous runner.
They are retained with their original configurations and are not results from
this single-split protocol.

Pinned dependencies were recorded on macOS with Python 3.13. Seeds and
PyTorch deterministic operations support repeat runs in the same environment;
exact results across platforms, library versions, or devices are not guaranteed.
