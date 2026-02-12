from preprocessors import get_detector, PREPROCESSOR_IDS, CALL_PARAMS


class ControlNetPreprocessorNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preprocessor": (PREPROCESSOR_IDS, {"default": "canny"}),
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

    def preprocess(self, image, preprocessor, resolution,
                   low_threshold=100, high_threshold=200):
        import torch
        import numpy as np
        from PIL import Image as PILImage

        detector = get_detector(preprocessor)
        call_kwargs = dict(CALL_PARAMS.get(preprocessor, {}))
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


class DiffSynthControlnetAdvancedNode:
    """QwenImageDiffsynthControlnet의 출력 MODEL을 받아
    스텝 범위 제어 + 선형 페이드를 적용하는 래퍼 노드."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength_start": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "strength_end": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "start_at": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "end_at": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "IXIWORKS/Image"

    def apply(self, model, strength_start, strength_end, start_at, end_at):
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_at)
        sigma_end = model_sampling.percent_to_sigma(end_at)

        str_begin = strength_start
        str_finish = strength_end
        use_fade = (strength_start != strength_end)

        uniform_scale = strength_start

        m = model.clone()
        existing_patches = m.model_options.get("patches", {})
        double_block_patches = existing_patches.get("double_block", [])

        wrapped = []
        for patch in double_block_patches:
            def make_wrapper(original_patch, s_start, s_end,
                             s_begin, s_finish, do_fade, u_scale):
                def wrapper(kwargs):
                    t_opts = kwargs.get("transformer_options")
                    if t_opts is not None:
                        sigmas = t_opts.get("sigmas")
                        if sigmas is not None:
                            sigma = sigmas[0].item()
                            if sigma > s_start or sigma < s_end:
                                return kwargs

                            if do_fade and s_start != s_end:
                                t = (s_start - sigma) / (s_start - s_end)
                                scale = s_begin + (s_finish - s_begin) * t
                            else:
                                scale = u_scale

                            if scale == 1.0:
                                return original_patch(kwargs)
                            if scale == 0.0:
                                return kwargs

                            before_img = kwargs["img"].clone()
                            result = original_patch(kwargs)
                            delta = result["img"] - before_img
                            result["img"] = before_img + delta * scale
                            return result

                    return original_patch(kwargs)
                return wrapper
            wrapped.append(make_wrapper(
                patch, sigma_start, sigma_end,
                str_begin, str_finish, use_fade, uniform_scale,
            ))

        if wrapped:
            m.model_options = m.model_options.copy()
            patches = m.model_options.get("patches", {}).copy()
            patches["double_block"] = wrapped
            m.model_options["patches"] = patches

        return (m,)


NODE_CLASS_MAPPINGS = {
    "ControlNetPreprocessor": ControlNetPreprocessorNode,
    "DiffSynthControlnetAdvanced": DiffSynthControlnetAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNetPreprocessor": "ControlNet Preprocessor",
    "DiffSynthControlnetAdvanced": "ControlNet Step Control",
}
