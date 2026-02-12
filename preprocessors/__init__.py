"""ControlNet preprocessors without controlnet_aux dependency."""

import torch

_detector_cache = {}


def _get_device():
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_detector(processor_id):
    """Get or create a cached detector instance. Lazy imports to avoid loading unused models."""
    if processor_id in _detector_cache:
        return _detector_cache[processor_id]

    device = _get_device()

    if processor_id == "canny":
        from .canny import CannyDetector
        detector = CannyDetector()

    elif processor_id == "depth":
        from .depth import MidasDetector
        detector = MidasDetector.from_pretrained()
        detector.model.to(device)

    elif processor_id == "lineart":
        from .lineart import LineartDetector
        detector = LineartDetector.from_pretrained()
        detector.model.to(device)

    elif processor_id == "pose":
        from .openpose import OpenposeDetector
        detector = OpenposeDetector.from_pretrained()
        detector.to(device)

    elif processor_id == "mlsd":
        from .mlsd import MLSDdetector
        detector = MLSDdetector.from_pretrained()
        detector.model.to(device)

    else:
        raise ValueError(f"Unknown preprocessor: {processor_id}")

    _detector_cache[processor_id] = detector
    return detector


PREPROCESSOR_IDS = ["canny", "depth", "lineart", "pose", "mlsd"]

CALL_PARAMS = {
    "canny": {},
    "depth": {},
    "lineart": {"coarse": False},
    "pose": {"hand_and_face": True},
    "mlsd": {},
}
