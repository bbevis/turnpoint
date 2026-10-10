"""Outcome-specific prediction changes when each turn's content is masked."""

import pandas as pd
import torch


def fit_mask(training, encoder):
    """Return the mean semantic vector across training turns, using fitted weights.

    Pass training conversations only, never validation or test conversations.
    Every turn has equal weight. The encoder's training/evaluation mode is restored.
    Save this tensor alongside the fitted model; recompute after refitting weights.
    """
    was_training = encoder.training
    encoder.eval()
    total, count = None, 0
    try:
        with torch.no_grad():
            for conversation in training:
                features = encoder(conversation["turns"])["semantic_features"]
                summed = features.sum(dim=0)
                total = summed if total is None else total + summed
                count += len(features)
        if not count:
            raise ValueError("Provide at least one training turn")
        mask = total / count
        if not torch.isfinite(mask).all():
            raise ValueError("Training representations contain non-finite values")
        return mask.detach().cpu()
    finally:
        encoder.train(was_training)


def score_turns(conversation, fitted, mask):
    """Mask each turn and return scores and trajectories as pandas DataFrames.

    fitted is the dictionary returned by train_model; mask comes from fit_mask.
    Only semantic features are replaced: length, speaker, and position remain.
    Predictions use original units for continuous outcomes and probabilities for
    classification. Binary probabilities refer to transforms['classes'][1];
    multiclass outcomes have separate rows for each class. Labels are not read.

    A signed difference is original minus masked prediction. immediate is the
    difference at the focal turn, final is the difference at the last turn, and
    dynamic is the mean absolute difference from the focal turn through the end.
    Each trajectory row includes original, masked, and difference values.
    speaker_id identifies the focal speaker; participant_id identifies whose
    outcome is predicted (None for conversation outcomes).

    Uses one original and one model pass per turn, with both modules temporarily
    in eval mode. Full trajectories require quadratic output space in turn count.
    Later observed dialogue stays fixed. The training mean is a reference mask,
    not guaranteed to be neutral; check sensitivity to alternative fitted masks.
    """
    encoder, model = fitted["encoder"], fitted["model"]
    encoder_mode, model_mode = encoder.training, model.training
    encoder.eval(); model.eval()
    turns = conversation["turns"]
    speakers = [turn["speaker_id"] for turn in turns]
    scores, trajectories = [], []

    def predictions(vectors):
        output = model(vectors, speakers)
        result = {}
        for scope in ("conversation_outcomes", "individual_outcomes"):
            groups = [(None, output[scope])] if scope == "conversation_outcomes" else output[scope].items()
            for participant, outcomes in groups:
                for name, values in outcomes.items():
                    transform = fitted["transforms"][scope][name]
                    kind = transform["kind"]
                    if kind == "continuous":
                        values = values * transform["scale"] + transform["mean"]
                        classes = [None]
                    elif kind == "binary":
                        values = values.sigmoid()
                        classes = [transform["classes"][1]]
                    else:
                        values = values.softmax(dim=-1)
                        classes = transform["classes"]
                    for column, label in enumerate(classes):
                        result[(scope, participant, name, label)] = values[:, column].cpu()
        return result

    try:
        with torch.no_grad():
            vectors = encoder(turns)["vectors"]
            mask = torch.as_tensor(mask, device=vectors.device, dtype=vectors.dtype)
            if mask.shape != (vectors.shape[1] - 2,) or not torch.isfinite(mask).all():
                raise ValueError("mask must be a finite semantic vector matching the encoder width")
            original = predictions(vectors)
            for i, turn in enumerate(turns):
                masked_vectors = vectors.clone()
                masked_vectors[i, :-2] = mask
                masked = predictions(masked_vectors)
                for (scope, participant, name, label), baseline in original.items():
                    changed = masked[(scope, participant, name, label)]
                    difference = baseline[i:] - changed[i:]
                    identity = {"conversation_id": conversation["conversation_id"],
                                "turn_number": turn["turn_number"], "speaker_id": turn["speaker_id"],
                                "scope": scope, "participant_id": participant,
                                "outcome": name, "class_label": label}
                    scores.append({**identity, "immediate": difference[0].item(),
                                   "dynamic": difference.abs().mean().item(),
                                   "final": difference[-1].item()})
                    for t in range(i, len(turns)):
                        trajectories.append({**identity, "prediction_turn_number": turns[t]["turn_number"],
                                             "original": baseline[t].item(), "masked": changed[t].item(),
                                             "difference": (baseline[t] - changed[t]).item()})
        return {"scores": pd.DataFrame(scores), "trajectories": pd.DataFrame(trajectories)}
    finally:
        encoder.train(encoder_mode); model.train(model_mode)
