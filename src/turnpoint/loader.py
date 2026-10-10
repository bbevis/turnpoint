"""Load TurnPoint's standard CSV format, independent of the source dataset."""

from itertools import groupby
from pathlib import Path

import pandas as pd


def load_conversations(folder):
    """Load conversations and user-defined targets from standard CSV files.

    Parameters
    ----------
    folder : str or pathlib.Path
        Directory containing ``turns.csv`` and at least one of
        ``conversation_outcomes.csv`` and ``individual_outcomes.csv``.
        Turns require conversation_id, turn_number, speaker_id, and text.
        Conversation outcomes use conversation_id as their unique key;
        individual outcomes use (conversation_id, speaker_id). All remaining
        outcome columns are targets. Extra turn columns are retained.

    Outputs
    ------
    dict
        ``conversation_id``: original ID, read as a string.
        ``turns``: list of row dictionaries sorted by numeric turn_number.
        ``conversation_outcomes``: outcome names mapped to values.
        ``individual_outcomes``: speaker IDs mapped to outcome dictionaries.
        Blank outcome cells become None; absent outcome rows become {}.
        Outcome column types are inferred by pandas. No outcomes are imputed.
    """
    folder = Path(folder)

    def read(name):
        return pd.read_csv(folder / name, dtype={"conversation_id": str, "speaker_id": str, "text": str},
                           keep_default_na=False, na_values=[""])

    turns = read("turns.csv")
    required = ["conversation_id", "turn_number", "speaker_id", "text"]
    if turns[required].isna().any().any():
        raise ValueError("Required turn fields cannot be blank")
    turns["turn_number"] = pd.to_numeric(turns["turn_number"], errors="raise")
    if (turns["turn_number"] < 1).any() or (turns["turn_number"] % 1 != 0).any():
        raise ValueError("turn_number must contain positive integers")
    if turns.duplicated(["conversation_id", "turn_number"]).any():
        raise ValueError("Turn numbers must be unique within each conversation")
    if not turns["text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Turn text must be nonblank strings")

    def outcomes(name, keys):
        if not (folder / name).exists():
            return {}
        table = read(name)
        if table[keys].isna().any().any() or table.duplicated(keys).any():
            raise ValueError(f"{name}: rows need nonblank, unique keys {keys}")
        if len(table.columns) == len(keys):
            raise ValueError(f"{name}: include at least one outcome column")
        valid = turns[keys].drop_duplicates().set_index(keys).index
        if not table.set_index(keys).index.isin(valid).all():
            raise ValueError(f"{name}: outcome keys do not match the turns")
        table = table.astype(object).where(pd.notna(table), None)
        return table.set_index(keys).to_dict(orient="index")

    if not any((folder / name).exists() for name in ("conversation_outcomes.csv", "individual_outcomes.csv")):
        raise ValueError("Provide conversation_outcomes.csv or individual_outcomes.csv")
    shared = outcomes("conversation_outcomes.csv", ["conversation_id"])
    individual = outcomes("individual_outcomes.csv", ["conversation_id", "speaker_id"])
    ordered = turns.sort_values(["conversation_id", "turn_number"])
    rows = (dict(zip(ordered.columns, row)) for row in ordered.itertuples(index=False, name=None))
    for conversation_id, group in groupby(rows, key=lambda row: row["conversation_id"]):
        conversation_turns = list(group)
        yield {
            "conversation_id": conversation_id,
            "turns": conversation_turns,
            "conversation_outcomes": shared.get(conversation_id, {}),
            "individual_outcomes": {
                speaker: individual.get((conversation_id, speaker), {})
                for speaker in dict.fromkeys(row["speaker_id"] for row in conversation_turns)
            },
        }
