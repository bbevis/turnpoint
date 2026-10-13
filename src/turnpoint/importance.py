"""Turn importance measured as consecutive prefix prediction updates."""

import pandas as pd
import torch


def score_turns(conversation, fitted):
    """Return one row per turn, outcome, participant, and classification class.

    Predict the final outcome after each observed turn. prediction_change is
    the current prediction minus the previous turn's prediction; importance
    uses training-standard-deviation units for continuous outcomes.
    importance_original is the absolute update in original units. Updates and
    importance are missing for the first turn, which has no
    preceding prefix. Continuous predictions use original outcome units;
    classification uses probabilities (binary: second training class).
    speaker_id owns the turn; participant_id owns the predicted outcome.
    No outcome labels are read. Uses one model pass in evaluation mode and
    restores the original module modes. These updates are not causal effects.
    """
    encoder, model = fitted["encoder"], fitted["model"]
    encoder_mode, model_mode = encoder.training, model.training
    encoder.eval()
    model.eval()
    turns = conversation["turns"]
    rows = []
    try:
        with torch.no_grad():
            output = model(encoder(turns)["vectors"], [t["speaker_id"] for t in turns])
            for scope in ("conversation_outcomes", "individual_outcomes"):
                groups = [(None, output[scope])] if scope == "conversation_outcomes" else output[scope].items()
                for participant, outcomes in groups:
                    for name, values in outcomes.items():
                        transform = fitted["transforms"][scope][name]
                        if transform["kind"] == "continuous":
                            values = values * transform["scale"] + transform["mean"]
                            classes = [None]
                        elif transform["kind"] == "binary":
                            values = values.sigmoid()
                            classes = [transform["classes"][1]]
                        else:
                            values = values.softmax(dim=-1)
                            classes = transform["classes"]
                        values = values.cpu()
                        for column, label in enumerate(classes):
                            for i, turn in enumerate(turns):
                                prediction = values[i, column].item()
                                previous = values[i - 1, column].item() if i else None
                                change = prediction - previous if i else None
                                rows.append({"conversation_id": conversation["conversation_id"],
                                    "turn_number": turn["turn_number"], "speaker_id": turn["speaker_id"],
                                    "scope": scope, "participant_id": participant,
                                    "outcome": name, "class_label": label,
                                    "prediction": prediction, "previous_prediction": previous,
                                    "prediction_change": change,
                                    "importance": abs(change) if change is not None else None})
        return standardize_importance(pd.DataFrame(rows), fitted["transforms"])
    finally:
        encoder.train(encoder_mode)
        model.train(model_mode)


def standardize_importance(scores, transforms):
    """Express continuous importance in training-set standard deviation units.

    Predictions and prediction_change retain original units. Classification
    importance remains a probability change. Training uses scale=1 when an
    outcome has zero variance; that same fallback applies here. Never estimate
    scales from test data. Safe to apply again to previously converted scores.
    """
    scores = scores.copy()
    scales = [transforms[scope][name].get("scale", 1.0)
              for scope, name in zip(scores.scope, scores.outcome)]
    scores["importance_original"] = scores.prediction_change.abs()
    scores["standardized_prediction_change"] = scores.prediction_change / scales
    scores["importance"] = scores.standardized_prediction_change.abs()
    scores["importance_units"] = [
        "training_standard_deviation" if transforms[scope][name]["kind"] == "continuous"
        else "probability" for scope, name in zip(scores.scope, scores.outcome)]
    return scores


def summarise_scores(scores):
    """Summarise scores separately by outcome, class, and target relation.

    Candidates have positive importance at or above the empirical 95th percentile
    within their outcome/relation/class group. Ties are retained; zero updates
    are never candidates. First-turn scores are excluded from all statistics.
    Primary candidate exports include shared and counterpart outcomes, not own
    outcomes. These are screening flags, not independently validated turning points.
    """
    scores = scores.copy()
    scores["target_relation"] = "conversation"
    individual = scores["scope"].eq("individual_outcomes")
    scores.loc[individual, "target_relation"] = "counterpart"
    scores.loc[individual & scores.speaker_id.eq(scores.participant_id), "target_relation"] = "own"
    rows, candidates = [], []
    keys = ["scope", "outcome", "target_relation", "class_label"]
    for identity, group in scores.groupby(keys, dropna=False):
        eligible = group.dropna(subset=["importance"])
        if eligible.empty:
            continue
        values = eligible.importance
        threshold = values.quantile(0.95)
        flagged = eligible[(values >= threshold) & (values > 0)].copy()
        row = dict(zip(keys, identity))
        row.update(conversations=group.conversation_id.nunique(), score_records=len(group),
                   eligible_updates=len(eligible), undefined_first_turn=len(group)-len(eligible),
                   unique_eligible_turns=len(eligible[["conversation_id", "turn_number"]].drop_duplicates()),
                   mean=values.mean(), sd=values.std(ddof=1), median=values.median(),
                   q25=values.quantile(.25), q75=values.quantile(.75), minimum=values.min(), maximum=values.max(),
                   positive_updates=int((eligible.prediction_change > 0).sum()),
                   negative_updates=int((eligible.prediction_change < 0).sum()),
                   zero_updates=int((eligible.prediction_change == 0).sum()),
                   candidate_threshold=threshold, candidate_flags=len(flagged),
                   candidate_unique_turns=len(flagged[["conversation_id", "turn_number"]].drop_duplicates()))
        rows.append(row)
        if identity[2] != "own":
            flagged["candidate_threshold"] = threshold
            candidates.append(flagged)
    summary = pd.DataFrame(rows)
    candidates = pd.concat(candidates, ignore_index=True) if candidates else scores.iloc[:0].copy()
    primary = scores[scores.target_relation.ne("own")]
    coverage = {"conversations_scored": int(scores.conversation_id.nunique()),
                "unique_turns_scored": len(scores[["conversation_id", "turn_number"]].drop_duplicates()),
                "unique_eligible_turns": len(primary.dropna(subset=["importance"])[["conversation_id", "turn_number"]].drop_duplicates()),
                "first_turns_without_updates": int(scores.conversation_id.nunique()),
                "score_records": len(scores), "undefined_score_records": int(scores.importance.isna().sum()),
                "importance_units": "continuous: training standard deviation; classification: probability",
                "candidate_rule": "importance > 0 and >= group 95th percentile; ties retained; empirical screening, not a significance test",
                "primary_relations": ["conversation", "counterpart"],
                "candidate_turn_outcome_flags": len(candidates),
                "candidate_unique_turns": len(candidates[["conversation_id", "turn_number"]].drop_duplicates())}
    return summary, candidates, coverage
