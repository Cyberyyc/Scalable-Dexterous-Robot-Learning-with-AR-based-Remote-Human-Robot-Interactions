"""Distance, progress, event, and hand-pose rewards (paper Eqs. 4–7)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RewardParameters:
    progress_weight: float = 1000.0
    event_weight: float = 1.0
    pose_weight: float = 1.0
    success: float = 1000.0
    collision: float = 100.0
    contact: float = 60.0
    pose_below: float = 7.0
    pose_above: float = 80.0
    alignment_threshold: float = 0.75


def alignment_cosine(hand_normal, hand_position, object_position):
    normal = np.asarray(hand_normal, dtype=float)
    direction = np.asarray(object_position, dtype=float) - np.asarray(hand_position, dtype=float)
    denominator = np.linalg.norm(normal) * np.linalg.norm(direction)
    if denominator <= np.finfo(float).eps:
        raise ValueError("Alignment requires a nonzero normal and hand-to-object direction.")
    return float(np.clip(np.dot(normal, direction) / denominator, -1.0, 1.0))


def manipulation_reward(distance, previous_distance, cosine_alignment,
                        *, success=False, collision=False, contact=False,
                        parameters=RewardParameters()):
    event = parameters.success if success else -parameters.collision if collision else -parameters.contact if contact else 0.0
    delta = cosine_alignment - parameters.alignment_threshold
    pose = (parameters.pose_below if delta <= 0 else parameters.pose_above) * delta
    return float(-distance + parameters.progress_weight * (previous_distance-distance)
                 + parameters.event_weight * event + parameters.pose_weight * pose)
