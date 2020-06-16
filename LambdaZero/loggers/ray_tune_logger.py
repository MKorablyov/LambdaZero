from typing import Dict


class RayTuneLogger:
    """
    Base Class to manage logging within a ray.tune trainable object.
    This implementation does nothing.
    """

    def __init__(self, config: Dict[str, str], log_dir: str, trial_id: str):
        pass

    def log_metrics(self, result_dict: Dict, step: int):
        pass


