"""Outcome training with frozen MPNet turn embeddings."""

from copy import deepcopy
from math import isfinite
from statistics import fmean, pstdev

import torch
from torch.nn import functional as F

from .encoder import TurnEncoder, MODEL_NAME, MODEL_REVISION
from .model import ConversationModel


def train_model(training, validation, conversation_outcomes=None, individual_outcomes=None, *,
                epochs=10, learning_rate=0.001, outcome_weights=None, encoder_options=None,
                model_options=None, seed=0, device="cpu",
                checkpoint=None):
    """Train the conversation Transformer using frozen cached turn embeddings.

    Splits must contain distinct conversation IDs. Continuous targets are scaled
    on training data only; missing outcomes are skipped. Loss averages over turns
    and weighted observed targets, then equally over labelled conversations.
    Validation outcome loss selects the best epoch after all epochs run.
    encoder_options accepts cache_dir/encoding_device; model_options configures
    the conversation Transformer. Only that Transformer and its heads are trained.
    Checkpoints contain its weights, outcome transforms, encoder configuration,
    pinned pretrained revision, and training history.
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
    encoder = TurnEncoder(**encoder_options).to(device)
    encoder.prepare(t["text"] for c in training + validation for t in c["turns"])
    model = ConversationModel(encoder.output_dim,
                              specifications["conversation_outcomes"], specifications["individual_outcomes"],
                              **model_options).to(device)
    parameters = list(model.parameters())
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
        return outcome_loss, outcome_loss

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
            best = deepcopy({"model": model.state_dict()})

    model.load_state_dict(best["model"])
    if checkpoint is not None:
        torch.save({**best, "text_model": MODEL_NAME, "text_model_revision": MODEL_REVISION, "transforms": transforms,
                    "specifications": specifications, "encoder_options": encoder_options,
                    "model_options": model_options, "best_epoch": best_epoch,
                    "best_validation_loss": best_loss, "history": history}, checkpoint)
    return {"encoder": encoder, "model": model,
            "transforms": transforms, "history": history,
            "best_epoch": best_epoch, "best_validation_loss": best_loss}


def load_model(path, device="cpu"):
    """Load a frozen-MPNet checkpoint for scoring without retraining.

    Reuses the exact pretrained revision and target transformations. Older GRU
    checkpoints are incompatible. Load only checkpoints from trusted sources.
    """
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("text_model") != MODEL_NAME or saved.get("text_model_revision") != MODEL_REVISION:
        raise ValueError("Checkpoint does not use the supported pinned MPNet encoder")
    encoder = TurnEncoder(**saved["encoder_options"]).to(device)
    model = ConversationModel(encoder.output_dim, **saved["specifications"], **saved["model_options"]).to(device)
    model.load_state_dict(saved["model"])
    encoder.eval(); model.eval()
    return {"encoder": encoder, "model": model, "transforms": saved["transforms"]}
