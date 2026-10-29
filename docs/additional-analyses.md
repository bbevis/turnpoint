# Additional analyses

Follow the [README](../README.md) to install TurnPoint and run HFC first.
Run the commands below from the repository root with the virtual environment active.
Use a new or empty output directory for each run.


These analyses reuse the same workflow but train separate models. They do not test
whether an HFC-trained model transfers to another dataset.

For IDE runs, edit the **EDIT THESE SETTINGS** blocks near the top of the scripts,
save, and click Run without script arguments. Select the repository's `.venv`
interpreter; paths are resolved from the repository root automatically.

### MIT Rental

Rental source workbooks are **not included in the repository**. To reproduce this
analysis, obtain the four `*_rental_wide_outcomes_deid.xlsx` workbooks and place them
in `data/MIT AI negotiation competition/outcomes/`. Without these inputs, skip Rental;
cloning the repository is sufficient for HFC, but not Rental.

In an IDE, set `EXERCISE = "rental"` and `DESTINATION = "data/rental_formatted"`
in `test/prepare_mit.py`. Run it, then set `DATA = "data/rental_formatted"` and
`OUTPUT = "output/rental_full"` in `run.py`, keeping the HFC outcome names.
Use `SAMPLE_SIZE = 0` and `CHECKPOINT = None`, then run it.

In the terminal:

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

In an IDE, run `test/prepare_casino.py` with its default settings. In `run.py`, set:

```python
DATA = "data/casino_formatted"
CONVERSATION_OUTCOMES = ["joint_points"]
INDIVIDUAL_OUTCOMES = ["points_scored", "satisfaction", "opponent_likeness"]
OUTPUT = "output/casino_full"
SAMPLE_SIZE = 0
CHECKPOINT = None
```

Save and run it. In the terminal:

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

In an IDE, set `DATA = "data/formatted"`,
`CHECKPOINT = "output/hfc_mpnet/model.pt"`, `OUTPUT = "output/hfc_scored"`, and
`SAMPLE_SIZE = 0` in `run.py`. Run it, then open and run
`test/summarise_examples.py`. Its `RUNS` setting lists the result folders to combine.
Restore `CHECKPOINT = None` before training another model.

In the terminal:

```bash
python run.py --data data/formatted --checkpoint output/hfc_mpnet/model.pt --output output/hfc_scored
python test/summarise_examples.py
```

Checkpoint scoring verifies the formatted input checksums and uses the saved test
split. The summary script combines HFC, Rental, and CaSiNo results into
`output/proposal_results/dataset_summary.csv`, `importance_summary.csv`, and
`prediction_metrics.csv`. Missing dataset runs are marked `pending`; the summary
can therefore also be generated after completing HFC alone.
