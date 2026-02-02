PREPROCESSOR_REGISTRY = {
    "canny": {
        "class": "CannyDetector",
        "from_pretrained": False,
        "call_params": {},
    },
    "depth": {
        "class": "MidasDetector",
        "from_pretrained": True,
        "call_params": {},
    },
    "lineart": {
        "class": "LineartDetector",
        "from_pretrained": True,
        "call_params": {"coarse": False},
    },
    "pose": {
        "class": "OpenposeDetector",
        "from_pretrained": True,
        "call_params": {"hand_and_face": True},
    },
    "mlsd": {
        "class": "MLSDdetector",
        "from_pretrained": True,
        "call_params": {},
    },
}

_DETECTOR_CLASSES = None


def _load_detector_classes():
    global _DETECTOR_CLASSES
    if _DETECTOR_CLASSES is not None:
        return _DETECTOR_CLASSES

    from controlnet_aux import (
        CannyDetector,
        LineartDetector,
        MidasDetector,
        MLSDdetector,
        OpenposeDetector,
    )

    _DETECTOR_CLASSES = {
        "CannyDetector": CannyDetector,
        "MidasDetector": MidasDetector,
        "LineartDetector": LineartDetector,
        "OpenposeDetector": OpenposeDetector,
        "MLSDdetector": MLSDdetector,
    }
    return _DETECTOR_CLASSES


class ControlNetPreprocessorNode:
    _detector_cache = {}

    PREPROCESSOR_IDS = list(PREPROCESSOR_REGISTRY.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preprocessor": (cls.PREPROCESSOR_IDS, {"default": "canny"}),
                "resolution": ("INT", {
                    "default": 512, "min": 256, "max": 2048, "step": 64,
                }),
            },
            "optional": {
                "low_threshold": ("INT", {
                    "default": 100, "min": 0, "max": 255, "step": 1,
                }),
                "high_threshold": ("INT", {
                    "default": 200, "min": 0, "max": 255, "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preprocess"
    CATEGORY = "IXIWORKS/Image"

    @classmethod
    def _get_detector(cls, processor_id):
        config = PREPROCESSOR_REGISTRY[processor_id]
        class_name = config["class"]

        if class_name not in cls._detector_cache:
            classes = _load_detector_classes()
            detector_cls = classes[class_name]

            try:
                if config["from_pretrained"]:
                    detector = detector_cls.from_pretrained(
                        "lllyasviel/Annotators"
                    )
                else:
                    detector = detector_cls()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load '{processor_id}' model: {e}\n"
                    f"Models auto-download from HuggingFace on first use. "
                    f"Check your internet connection."
                ) from e

            cls._detector_cache[class_name] = detector

        return cls._detector_cache[class_name]

    def preprocess(self, image, preprocessor, resolution,
                   low_threshold=100, high_threshold=200):
        import torch
        import numpy as np
        from PIL import Image as PILImage

        config = PREPROCESSOR_REGISTRY[preprocessor]
        detector = self._get_detector(preprocessor)

        call_kwargs = dict(config["call_params"])
        call_kwargs["detect_resolution"] = resolution
        call_kwargs["image_resolution"] = resolution

        if preprocessor == "canny":
            call_kwargs["low_threshold"] = low_threshold
            call_kwargs["high_threshold"] = high_threshold

        results = []
        for i in range(image.shape[0]):
            pil_img = PILImage.fromarray(
                (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            )

            try:
                processed = detector(pil_img, **call_kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"ControlNet preprocessor '{preprocessor}' failed: {e}"
                ) from e

            if isinstance(processed, tuple):
                processed = processed[0]

            if isinstance(processed, PILImage.Image):
                processed = processed.convert("RGB")

            result = torch.from_numpy(
                np.array(processed).astype(np.float32) / 255.0
            )

            if result.dim() == 2:
                result = result.unsqueeze(-1).expand(-1, -1, 3)

            results.append(result)

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "ControlNetPreprocessor": ControlNetPreprocessorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNetPreprocessor": "ControlNet Preprocessor (Image)",
}
