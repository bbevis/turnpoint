"""Turn-level Transformer with masked self-attention and outcome prediction layers."""

import torch
from torch import nn


class ConversationModel(nn.Module):
    """Predict eventual outcomes using a Transformer with masked self-attention.

    Uses PyTorch TransformerEncoder layers: multi-head self-attention, a ReLU
    feed-forward network, residual connections, and post-sublayer layer norms.
    A future-turn mask restricts attention to current and preceding turns.
    Sinusoidal positions encode order. No pretrained weights or cross-attention
    are used; this model is trained for outcomes, not next-token generation.

    Parameters
    ----------
    input_dim : int
        Encoder vector width (sparse_dim + 2; 514 with the default encoder).
    conversation_outcomes, individual_outcomes : dict, optional
        Outcome names mapped to 'continuous', 'binary', or an integer class
        count >= 2. For example, {'satisfaction': 'continuous', 'choice': 3}.
        Individual layers are shared across participants, conditioned on whose
        outcome is predicted. At least one outcome must be supplied.
    hidden_dim, num_layers, num_heads, dropout : int or float
        Transformer settings; defaults are 128, 2, 4, and 0.1. hidden_dim must
        be divisible by num_heads. All weights start untrained.
    """

    def __init__(self, input_dim, conversation_outcomes=None, individual_outcomes=None,
                 hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.conversation_outcomes = dict(conversation_outcomes or {})
        self.individual_outcomes = dict(individual_outcomes or {})
        if not (self.conversation_outcomes or self.individual_outcomes):
            raise ValueError("Specify at least one outcome")

        def layers(specifications):
            result = []
            for kind in specifications.values():
                if kind in ("continuous", "binary"):
                    width = 1
                elif type(kind) is int and kind >= 2:
                    width = kind
                else:
                    raise ValueError("Outcome type must be continuous, binary, or a class count >= 2")
                result.append(nn.Linear(hidden_dim, width))
            return nn.ModuleList(result)

        self.conversation_layers = layers(self.conversation_outcomes)
        self.individual_layers = layers(self.individual_outcomes)
        self.projection = nn.Linear(input_dim, hidden_dim)
        # 0: shared conversation stream; 1: target speaks; 2: someone else speaks.
        self.target_speaker = nn.Embedding(3, hidden_dim)
        # PyTorch encoder blocks accept a self-attention mask; no decoder or
        # encoder-decoder cross-attention is needed for outcome prediction.
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim * 4,
                                       dropout=dropout, activation="relu",
                                       batch_first=True, norm_first=False, bias=True),
            num_layers, enable_nested_tensor=False,
        )
        # PyTorch clones the initial layer; independently initialise its matrices.
        for layer in self.transformer.layers:
            for parameter in layer.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, vectors, speaker_ids):
        """Predict from one ordered conversation's encoder vectors and speaker IDs.

        vectors is a [turns, input_dim] tensor; speaker_ids has one original ID
        per turn in the same order. Returns conversation_outcomes[name] and
        individual_outcomes[speaker_id][name], each a [turns, output_width]
        tensor. Continuous outputs are raw scores; binary/multiclass outputs
        are logits (use sigmoid/softmax for probabilities). No labels enter.

        Attention is restricted to current and preceding turns.
        Sinusoidal positions encode turn order without a fixed length limit.
        Individual streams identify target-versus-other speakers, preserving
        arbitrary ID relabelling. In multi-party data, other speakers are pooled
        in this target indicator. Use eval() for deterministic inference.
        """
        speaker_ids = list(speaker_ids)
        if vectors.ndim != 2 or not len(vectors) or len(vectors) != len(speaker_ids):
            raise ValueError("Provide nonempty turn vectors and one speaker ID per turn")
        participants = list(dict.fromkeys(speaker_ids)) if self.individual_outcomes else []
        base = self.projection(vectors)
        turn_count, width = base.shape
        dimensions = torch.arange(width, device=base.device)
        angles = torch.arange(turn_count, device=base.device, dtype=base.dtype)[:, None] / (
            10000 ** (2 * torch.div(dimensions, 2, rounding_mode="floor") / width))
        positions = torch.where(dimensions % 2 == 0, angles.sin(), angles.cos())
        target_flags = [[0] * turn_count] + [
            [1 if speaker == target else 2 for speaker in speaker_ids] for target in participants
        ]
        flags = torch.tensor(target_flags, dtype=torch.long, device=base.device)
        inputs = (base + positions).unsqueeze(0) + self.target_speaker(flags)
        # True blocks attention: entries above the diagonal are future turns.
        # The diagonal remains visible because predictions include the current turn.
        future_turn_mask = torch.ones(turn_count, turn_count, dtype=torch.bool, device=base.device).triu(1)
        states = self.transformer(inputs, mask=future_turn_mask)

        def predict(state, names, layers):
            return {name: layer(state) for name, layer in zip(names, layers)}

        return {
            "conversation_outcomes": predict(states[0], self.conversation_outcomes, self.conversation_layers),
            "individual_outcomes": {
                participant: predict(states[index + 1], self.individual_outcomes, self.individual_layers)
                for index, participant in enumerate(participants)
            },
        }
