"""Joint training of the turn encoder and conversation outcome model."""

from copy import deepcopy
from math import isfinite
from statistics import fmean, pstdev

import torch
from torch.nn import functional as F

from .encoder import TurnEncoder, build_vocabulary
from .model import ConversationModel


def train_model(training, validation, conversation_outcomes=None, individual_outcomes=None, *,
                epochs=10, learning_rate=0.001, reconstruction_weight=0.1,
                sparsity_weight=0.001, outcome_weights=None, encoder_options=None,
                model_options=None, vocabulary_options=None, seed=0, device="cpu",
                checkpoint=None):
    """Fit both neural components using separate training/validation conversations.

    training, validation: iterables from load_conversations, with disjoint IDs.
    conversation_outcomes, individual_outcomes: name/type dictionaries matching
    ConversationModel (continuous, binary, or integer class count).
    epochs, learning_rate: Adam optimisation settings. One conversation per step.
    reconstruction_weight, sparsity_weight: nonnegative auxiliary loss weights.
    outcome_weights: optional scope/name weights, e.g.
        {'conversation_outcomes': {'score': 2}}; unspecified targets weigh 1.
    encoder_options, model_options, vocabulary_options: constructor/build options.
    seed, device: random seed and PyTorch device, default CPU.
    checkpoint: optional output file for the best weights and preprocessing.

    Returns a dictionary with encoder, model, vocabulary, transforms, history,
    best_epoch, and best_validation_loss. Returned components are in eval mode.
    Continuous targets are standardised; binary/multiclass outputs remain logits.
    Transforms and vocabulary are learned on training data only. Missing targets
    are skipped; all-missing conversations do not contribute to optimisation.
    Losses average over turns and observed weighted targets within a conversation,
    then equally over labelled conversations. Validation outcome loss alone
    selects the best epoch. Both input iterables are materialised in memory.
    """
    training, validation = list(training), list(validation)
    for dataset in (training, validation):
        ids = [c["conversation_id"] for c in dataset]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Each split needs conversations with unique IDs")
    if {c["conversation_id"] for c in training} & {c["conversation_id"] for c in validation}:
        raise ValueError("Training and validation conversation IDs must not overlap")
    if type(epochs) is not int or epochs < 1 or not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if any(not isfinite(w) or w < 0 for w in (reconstruction_weight, sparsity_weight)):
        raise ValueError("Auxiliary loss weights must be finite and nonnegative")
    specifications = {"conversation_outcomes": dict(conversation_outcomes or {}),
                      "individual_outcomes": dict(individual_outcomes or {})}
    outcome_weights = outcome_weights or {}
    weights = {scope: {name: outcome_weights.get(scope, {}).get(name, 1.0)
                      for name in names} for scope, names in specifications.items()}
    if any(not isfinite(w) or w < 0 for scope in weights.values() for w in scope.values()):
        raise ValueError("Outcome weights must be finite and nonnegative")

    def targets(conversation):
        for scope, names in specifications.items():
            groups = [(None, conversation[scope])] if scope == "conversation_outcomes" else conversation[scope].items()
            for participant, values in groups:
                for name in names:
                    if values.get(name) is not None:
                        yield scope, participant, name, values[name]

    observations = {scope: {name: [] for name in names} for scope, names in specifications.items()}
    for conversation in training:
        for scope, _, name, value in targets(conversation):
            observations[scope][name].append(value)
    transforms = {scope: {} for scope in specifications}
    for scope, names in specifications.items():
        for name, kind in names.items():
            values = observations[scope][name]
            if not values:
                raise ValueError(f"No training observations for {scope}.{name}")
            if kind == "continuous":
                values = [float(value) for value in values]
                if not all(isfinite(value) for value in values):
                    raise ValueError(f"Non-finite target: {name}")
                transform = {"kind": kind, "mean": fmean(values), "scale": pstdev(values) or 1.0}
            elif kind == "binary" or (type(kind) is int and kind >= 2):
                if any(isinstance(v, (int, float)) and not isfinite(v) for v in values):
                    raise ValueError(f"Non-finite target: {name}")
                classes = [0, 1] if kind == "binary" and set(values) <= {0, 1} else sorted(set(values), key=lambda v: (type(v).__name__, str(v)))
                if len(classes) != (2 if kind == "binary" else kind):
                    raise ValueError(f"{name}: training labels do not match the specified class count")
                transform = {"kind": kind, "classes": classes}
            else:
                raise ValueError(f"Unsupported outcome type: {kind}")
            transforms[scope][name] = transform

    def encode(scope, name, value):
        transform = transforms[scope][name]
        if transform["kind"] == "continuous":
            value = (float(value) - transform["mean"]) / transform["scale"]
            if not isfinite(value):
                raise ValueError(f"Non-finite target: {name}")
            return value
        if value not in transform["classes"]:
            raise ValueError(f"{name}: label {value!r} was not in the training classes")
        return transform["classes"].index(value)

    for dataset in (training, validation):
        labelled = 0
        for conversation in dataset:
            labelled += any(weights[s][n] > 0 for s, _, n, _ in targets(conversation))
            for scope, _, name, value in targets(conversation):
                encode(scope, name, value)
        if not labelled:
            raise ValueError("Each split needs at least one observed, positively weighted target")

    torch.manual_seed(seed)
    encoder_options, model_options = encoder_options or {}, model_options or {}
    vocabulary = build_vocabulary((t["text"] for c in training for t in c["turns"]), **(vocabulary_options or {}))
    encoder = TurnEncoder(vocabulary, **encoder_options).to(device)
    model = ConversationModel(encoder.sparse_encoder.out_features + 2,
                              specifications["conversation_outcomes"], specifications["individual_outcomes"],
                              **model_options).to(device)
    parameters = list(encoder.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def loss_for(conversation):
        observed = [(s, p, n, v) for s, p, n, v in targets(conversation) if weights[s][n] > 0]
        if not observed:
            return None
        encoded = encoder(conversation["turns"])
        predicted = model(encoded["vectors"], [t["speaker_id"] for t in conversation["turns"]])
        losses, total_weight = [], 0.0
        for scope, participant, name, value in observed:
            output = predicted[scope][name] if participant is None else predicted[scope][participant][name]
            target = encode(scope, name, value)
            kind = specifications[scope][name]
            if kind == "continuous":
                loss = F.mse_loss(output, torch.full_like(output, target))
            elif kind == "binary":
                loss = F.binary_cross_entropy_with_logits(output, torch.full_like(output, target))
            else:
                labels = torch.full((len(output),), target, dtype=torch.long, device=output.device)
                loss = F.cross_entropy(output, labels)
            weight = weights[scope][name]
            losses.append(weight * loss)
            total_weight += weight
        outcome_loss = torch.stack(losses).sum() / total_weight
        total = outcome_loss + reconstruction_weight * encoded["reconstruction_loss"] + sparsity_weight * encoded["sparsity_loss"]
        return outcome_loss, total

    history, best, best_loss, best_epoch = [], None, float("inf"), 0
    for epoch in range(1, epochs + 1):
        encoder.train(); model.train()
        training_losses = []
        for index in torch.randperm(len(training)).tolist():
            losses = loss_for(training[index])
            if losses is None:
                continue
            optimizer.zero_grad()
            if not torch.isfinite(losses[1]):
                raise ValueError("Non-finite training loss")
            losses[1].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            training_losses.append(losses[0].item())
        encoder.eval(); model.eval()
        with torch.no_grad():
            validation_losses = [losses[0].item() for c in validation if (losses := loss_for(c)) is not None]
        validation_loss = fmean(validation_losses)
        if not isfinite(validation_loss):
            raise ValueError("Non-finite validation loss")
        history.append({"epoch": epoch, "training_outcome_loss": fmean(training_losses),
                        "validation_outcome_loss": validation_loss})
        print(f"Epoch {epoch}/{epochs}: training={history[-1]['training_outcome_loss']:.4f}, validation={validation_loss:.4f}")
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best = deepcopy({"encoder": encoder.state_dict(), "model": model.state_dict()})

    encoder.load_state_dict(best["encoder"])
    model.load_state_dict(best["model"])
    if checkpoint is not None:
        torch.save({**best, "vocabulary": vocabulary, "transforms": transforms,
                    "specifications": specifications, "encoder_options": encoder_options,
                    "model_options": model_options, "best_epoch": best_epoch,
                    "best_validation_loss": best_loss, "history": history}, checkpoint)
    return {"encoder": encoder, "model": model, "vocabulary": vocabulary,
            "transforms": transforms, "history": history,
            "best_epoch": best_epoch, "best_validation_loss": best_loss}
