"""Train and evaluate TurnPoint with one 80/10/10 conversation split.

Example: python run.py --data data/formatted --conversation-outcomes value_created
Supply outcome names; continuous is the default. All model settings have defaults.
"""

import argparse
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch

from turnpoint import fit_mask, load_conversations, score_turns, train_model

def json_object(value):
    """Parse a command-line JSON dictionary, with a readable argument error."""
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("Provide a JSON object, e.g. {\"score\": \"continuous\"}")
    return result


def metrics(predictions):
    """Evaluate final-turn test predictions against training-only baselines.

    Continuous: MAE, RMSE, R-squared. Classification: accuracy and log loss.
    Missing observed targets are omitted; undefined R-squared stays blank.
    """
    rows = []
    for (scope, outcome), group in predictions.groupby(["scope", "outcome"]):
        group = group.dropna(subset=["observed"])
        if group.empty:
            continue
        continuous = group.iloc[0]["kind"] == "continuous"
        for estimator in ("model", "training_baseline"):
            row = {"scope": scope, "outcome": outcome, "estimator": estimator, "n": len(group)}
            if continuous:
                observed = group["observed"].to_numpy(dtype=float)
                predicted = group["predicted" if estimator == "model" else "baseline"].to_numpy(dtype=float)
                row.update(mae=mean_absolute_error(observed, predicted),
                    rmse=mean_squared_error(observed, predicted) ** 0.5,
                    r2=r2_score(observed, predicted) if len(group) > 1 and np.ptp(observed) > 0 else None)
            else:
                observed = group["observed_index"].to_numpy(dtype=int)
                column = "probabilities" if estimator == "model" else "baseline_probabilities"
                probabilities = np.array([json.loads(value) for value in group[column]])
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                row.update(accuracy=accuracy_score(observed, probabilities.argmax(axis=1)),
                           log_loss=log_loss(observed, probabilities, labels=np.arange(probabilities.shape[1])))
            rows.append(row)
    return pd.DataFrame(rows)


def run(args):
    """Train one model on 80%, select its epoch on 10%, and test on 10%.

    Whole conversations stay together. Vocabulary, scaling, baselines, and masks
    use training data only. Test data never selects epochs. Input tables are
    loaded into memory; model/training settings are internal defaults.
    """
    if not (args.conversation_outcomes or args.individual_outcomes):
        raise ValueError("Specify conversation-outcomes and/or individual-outcomes")
    if args.sample_size < 0 or args.epochs < 1 or args.score_limit < 0:
        raise ValueError("Use nonnegative sample/score sizes and positive epochs")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new output directory to avoid mixing different runs")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    conversations = list(load_conversations(args.data))
    indices = np.arange(len(conversations))
    if args.sample_size and args.sample_size < len(indices):
        indices, _ = train_test_split(indices, train_size=args.sample_size, random_state=args.seed)
        indices = np.sort(indices)
    total_conversations = len(conversations)
    selected = [conversations[i] for i in indices]
    del conversations
    if len(selected) < 10:
        raise ValueError("Provide at least 10 conversations for the 80/10/10 split")
    training, heldout = train_test_split(np.arange(len(selected)), test_size=0.2, random_state=args.seed)
    validation, test = train_test_split(heldout, test_size=0.5, random_state=args.seed)
    print(f"Using {len(selected):,} of {total_conversations:,} conversations: "
          f"{len(training):,} training, {len(validation):,} validation, {len(test):,} test; "
          f"{args.epochs} epochs; turn scores for up to {args.score_limit} test conversations.", flush=True)
    assignments = [{"role": role, "conversation_id": selected[i]["conversation_id"]}
                   for role, members in [("training", training), ("validation", validation), ("test", test)]
                   for i in members]
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(assignments).to_csv(args.output / "splits.csv", index=False)
    hashes = {}
    for name in ("turns.csv", "conversation_outcomes.csv", "individual_outcomes.csv"):
        if not (args.data / name).exists():
            continue
        digest = hashlib.sha256()
        with (args.data / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    encoder_options = {"embedding_dim": 100, "hidden_dim": 128, "sparse_dim": 512, **args.encoder_options}
    model_options = {"hidden_dim": 128, "num_layers": 2, "num_heads": 4, "dropout": 0.1, **args.model_options}
    vocabulary_options = {"min_count": 2, "max_words": 20000, **args.vocabulary_options}
    configuration = {**vars(args), "selected_conversations": len(selected), "input_sha256": hashes,
        "split_fractions": {"training": 0.8, "validation": 0.1, "test": 0.1},
        "split_counts": {"training": len(training), "validation": len(validation), "test": len(test)},
        "encoder_options": encoder_options, "model_options": model_options, "vocabulary_options": vocabulary_options,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in distributions()}}
    (args.output / "configuration.json").write_text(json.dumps(configuration, indent=2, default=str), encoding="utf-8")
    started = time.monotonic()
    predictions = []
    fitted = train_model([selected[i] for i in training], [selected[i] for i in validation],
        args.conversation_outcomes, args.individual_outcomes, epochs=args.epochs, seed=args.seed,
        learning_rate=args.learning_rate, reconstruction_weight=args.reconstruction_weight,
        sparsity_weight=args.sparsity_weight, outcome_weights=args.outcome_weights, device=args.device,
        encoder_options=encoder_options, model_options=model_options, vocabulary_options=vocabulary_options,
        checkpoint=args.output / "model.pt")
    pd.DataFrame(fitted["history"]).to_csv(args.output / "history.csv", index=False)
    summary = {"best_epoch": fitted["best_epoch"], "best_validation_loss": fitted["best_validation_loss"],
               "training_seconds": time.monotonic() - started}
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training finished in {summary['training_seconds'] / 60:.1f} minutes; selected epoch {summary['best_epoch']}.", flush=True)
    mask = fit_mask((selected[i] for i in training), fitted["encoder"]) if args.score_limit else None
    if mask is not None:
        torch.save(mask, args.output / "mask.pt")
    baselines = {}
    for scope, transforms in fitted["transforms"].items():
        for name, transform in transforms.items():
            if transform["kind"] == "continuous":
                continue
            counts = np.zeros(len(transform["classes"]))
            for index in training:
                targets = [selected[index][scope]] if scope == "conversation_outcomes" else selected[index][scope].values()
                for target in targets:
                    if target.get(name) is not None:
                        counts[transform["classes"].index(target[name])] += 1
            baselines[(scope, name)] = (counts / counts.sum()).tolist()
    with torch.no_grad():
        for position, index in enumerate(test):
            conversation = selected[index]
            turns = conversation["turns"]
            output = fitted["model"](fitted["encoder"](turns)["vectors"], [t["speaker_id"] for t in turns])
            for scope in ("conversation_outcomes", "individual_outcomes"):
                groups = [(None, output[scope])] if scope == "conversation_outcomes" else output[scope].items()
                for participant, outcomes in groups:
                    observed = conversation[scope] if participant is None else conversation[scope].get(participant, {})
                    for name, values in outcomes.items():
                        transform = fitted["transforms"][scope][name]
                        value = observed.get(name)
                        row = {"conversation_id": conversation["conversation_id"],
                            "scope": scope, "participant_id": participant,
                            "outcome": name, "kind": transform["kind"], "observed": value}
                        if transform["kind"] == "continuous":
                            row.update(predicted=values[-1, 0].item() * transform["scale"] + transform["mean"],
                                       baseline=transform["mean"])
                        else:
                            classes = transform["classes"]
                            if value is not None and value not in classes:
                                raise ValueError(f"{name}: test label absent from training classes")
                            if transform["kind"] == "binary":
                                probability = values[-1, 0].sigmoid().item()
                                probabilities = [1 - probability, probability]
                            else:
                                probabilities = values[-1].softmax(dim=-1).cpu().tolist()
                            baseline = baselines[(scope, name)]
                            row.update(predicted=classes[int(np.argmax(probabilities))],
                                baseline=classes[int(np.argmax(baseline))],
                                observed_index=classes.index(value) if value is not None else None,
                                classes=json.dumps(classes), probabilities=json.dumps(probabilities),
                                baseline_probabilities=json.dumps(baseline))
                        predictions.append(row)
            if position < args.score_limit:
                importance = score_turns(conversation, fitted, mask)
                for name, table in importance.items():
                    table.to_csv(args.output / f"{name}.csv", mode="a", header=position == 0, index=False)
    table = pd.DataFrame(predictions)
    table.to_csv(args.output / "predictions.csv", index=False)
    metrics(table).to_csv(args.output / "metrics.csv", index=False)
    print(f"Completed training and test evaluation. Results: {args.output}", flush=True)


def outcome_definitions(names):
    """Accept plain outcome names, optional name:binary/name:3, or legacy JSON."""
    if len(names) == 1 and names[0].lstrip().startswith("{"):
        return json_object(names[0])
    definitions = {}
    for item in names:
        name, separator, kind = item.rpartition(":")
        if not separator:
            name, kind = item, "continuous"
        elif kind.isdigit() and int(kind) >= 2:
            kind = int(kind)
        elif kind not in ("continuous", "binary"):
            raise argparse.ArgumentTypeError("Use an outcome name, name:binary, or name:3 for three classes")
        if not name or name in definitions:
            raise argparse.ArgumentTypeError("Outcome names must be nonblank and unique within each scope")
        definitions[name] = kind
    return definitions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog="Model and training settings use fixed internal defaults.")
    parser.add_argument("--data", type=Path, required=True, help="Folder containing your formatted CSVs")
    parser.add_argument("--conversation-outcomes", nargs="+", default=[], metavar="NAME", help="Conversation outcome names (continuous by default)")
    parser.add_argument("--individual-outcomes", nargs="+", default=[], metavar="NAME", help="Individual outcome names (continuous by default)")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional conversation limit (default: all; 0: all)")
    parser.add_argument("--output", type=Path, default=Path("output/run"), help="Results folder (default: output/run; must be new or empty)")
    # Internal settings, not command-line options.
    parser.set_defaults(
        epochs=10, seed=0, score_limit=10, encoder_options={}, model_options={}, vocabulary_options={},
        outcome_weights={}, learning_rate=0.001, reconstruction_weight=0.1,
        sparsity_weight=0.001, device="cpu",
    )
    args = parser.parse_args()
    try:
        args.conversation_outcomes = outcome_definitions(args.conversation_outcomes)
        args.individual_outcomes = outcome_definitions(args.individual_outcomes)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if not (args.conversation_outcomes or args.individual_outcomes):
        parser.error("Specify at least one conversation or individual outcome")
    run(args)
