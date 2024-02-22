from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """
    Configuration for training.

    Attributes:
        freeze_nc: whether to set noncanonicals to 5 and freeze them
        loop_pen: whether to use the extended nussinov model with hairpin and internal loop penalties
        h: minimum loop length
        lr: Learning rate for the optimizer
        steps: Total number of training steps
        examples_per_step: Number of training examples per step
        val_examples_per_step: Number of validation examples per step
        patience_warmup: Number of warm-up steps before counting patience
        patience: Number of steps to wait for improvement before early stopping
    """
    freeze_nc: bool = False
    loop_pen: bool = False
    h: int = 4
    lr: float = 0.01
    steps: int = 7000
    examples_per_step: int = 100
    val_examples_per_step: int = 100
    patience_warmup: int = 100
    patience: int = 50


@dataclass(frozen=True)
class LossConfig:
    """
    Configuration for loss function parameters.

    Attributes:
        l2_reg: Weight for L2 regularization on energy values.
        entropy_weight: Weight for entropy penalty on predicted unpaired probabilities.
        mag_weight: Weight for mean magnitude alignment (avg(pred) vs avg(target)).
        clip_target: Whether to clip target reactivities to [0, 1] during training.
        downweight_clipped: Multiplier for loss at clipped target positions.
        noncanon_weight: Weight for penalizing favorable non-canonical base pairs.
        canon_weight: Weight for penalizing unfavorable canonical base pairs.
        canon_usage_weight: Weight for encouraging use of canonical base pairs in structure.
        helix_weight: Weight for encouraging stem-like (helix) structures in the predicted BPP.
        eps: Small constant to prevent log(0) or division errors.
    """
    l2_reg: float = 1e-4
    entropy_weight: float = 0.0
    mag_weight: float = 0.0
    clip_target: bool = True
    downweight_clipped: float = 0.5
    noncanon_weight: float = 0.1
    canon_weight: float = 0.1
    canon_usage_weight: float = 0.0
    helix_weight: float = 0.0
    eps: float = 1e-6
