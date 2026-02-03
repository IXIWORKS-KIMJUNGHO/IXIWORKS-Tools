class AnyType(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


ANY = AnyType("*")


class SwitchBooleanNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_false": (ANY, {"lazy": True}),
                "on_true": (ANY, {"lazy": True}),
                "boolean_switch": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    CATEGORY = "IXIWORKS/Utils"

    def check_lazy_status(self, on_false, on_true, boolean_switch):
        needed = "on_true" if boolean_switch else "on_false"
        if (boolean_switch and on_true is None) or (not boolean_switch and on_false is None):
            return [needed]
        return []

    def switch(self, on_false, on_true, boolean_switch):
        return (on_true if boolean_switch else on_false,)


class StringToListNode:
    MAX_INPUTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "count": ("INT", {"default": 4, "min": 1, "max": cls.MAX_INPUTS, "step": 1}),
            "prompt_1": ("STRING", {"default": "", "multiline": True}),
        }
        optional = {
            f"prompt_{i}": ("STRING", {"default": "", "multiline": True})
            for i in range(2, cls.MAX_INPUTS + 1)
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("strings",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert"
    CATEGORY = "IXIWORKS/Utils"

    def convert(self, count, **kwargs):
        result = []
        for i in range(1, count + 1):
            key = f"prompt_{i}"
            value = kwargs.get(key, "").strip()
            if value:
                result.append(value)
        if not result:
            result.append("")
        return (result,)


class JoinStringsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_a": ("STRING", {"forceInput": True}),
                "string_b": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "separator": ("STRING", {"default": " "}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("joined_string",)
    FUNCTION = "join"
    CATEGORY = "IXIWORKS/Utils"

    def join(self, string_a, string_b, separator=" "):
        return (f"{string_a}{separator}{string_b}",)


class SwitchCaseNode:
    MAX_INPUTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "count": ("INT", {"default": 3, "min": 2, "max": cls.MAX_INPUTS, "step": 1}),
            "select": ("INT", {"default": 0, "min": 0, "max": cls.MAX_INPUTS - 1, "step": 1}),
        }
        optional = {
            f"input_{i}": (ANY, {"lazy": True})
            for i in range(cls.MAX_INPUTS)
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    CATEGORY = "IXIWORKS/Utils"

    def check_lazy_status(self, count, select, **kwargs):
        index = max(0, min(select, count - 1))
        key = f"input_{index}"
        if kwargs.get(key) is None:
            return [key]
        return []

    def switch(self, count, select, **kwargs):
        index = max(0, min(select, count - 1))
        key = f"input_{index}"
        return (kwargs.get(key, None),)


class SaveTextNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename": ("STRING", {"default": "output.txt"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "IXIWORKS/Utils"
    OUTPUT_NODE = True

    def save(self, text, filename):
        import os
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        file_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        return {}


class LoadImageListNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filenames": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load"
    CATEGORY = "IXIWORKS/Utils"

    def load(self, filenames):
        import os
        import numpy as np
        import torch
        from PIL import Image
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        names = [n.strip() for n in filenames.split(",") if n.strip()]

        images = []
        for name in names:
            path = os.path.join(input_dir, name)
            img = Image.open(path).convert("RGB")
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            images.append(img_tensor)

        if not images:
            blank = torch.zeros(1, 64, 64, 3)
            images.append(blank)

        return (images,)


class ImageToListNode:
    MAX_INPUTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        # Only count is required, images are added dynamically via JS
        required = {
            "count": ("INT", {"default": 4, "min": 1, "max": cls.MAX_INPUTS, "step": 1}),
        }
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert"
    CATEGORY = "IXIWORKS/Utils"

    def convert(self, count, **kwargs):
        result = []
        for i in range(1, count + 1):
            key = f"image_{i}"
            img = kwargs.get(key)
            if img is not None:
                result.append(img)
        if not result:
            import torch
            result.append(torch.zeros(1, 64, 64, 3))
        return (result,)


ASPECT_RATIOS = {
    "21:9": (21, 9),
    "1.85:1": (1.85, 1),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
}


class EmptyLatentRatioNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio": (list(ASPECT_RATIOS.keys()), {"default": "16:9"}),
                "long_side": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 64,
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "IXIWORKS/Utils"

    def generate(self, ratio, long_side, batch_size):
        import torch

        w_ratio, h_ratio = ASPECT_RATIOS[ratio]

        if w_ratio >= h_ratio:
            width = long_side
            height = int(long_side * h_ratio / w_ratio)
        else:
            height = long_side
            width = int(long_side * w_ratio / h_ratio)

        # Ensure divisible by 8 (latent space requirement)
        width = (width // 8) * 8
        height = (height // 8) * 8

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples": latent}, width, height)


class BypassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bypass": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input": (ANY,),
            }
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "IXIWORKS/Utils"

    def execute(self, bypass, **kwargs):
        return (kwargs.get("input", None),)


NODE_CLASS_MAPPINGS = {
    "SwitchBoolean": SwitchBooleanNode,
    "StringToList": StringToListNode,
    "ConcatStrings": JoinStringsNode,
    "SwitchCase": SwitchCaseNode,
    "SaveText": SaveTextNode,
    "LoadImageList": LoadImageListNode,
    "ImageToList": ImageToListNode,
    "Bypass": BypassNode,
    "EmptyLatentRatio": EmptyLatentRatioNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwitchBoolean": "Switch (Utils)",
    "StringToList": "String to List (Utils)",
    "ConcatStrings": "Concat Strings (Utils)",
    "SwitchCase": "Switch Case (Utils)",
    "SaveText": "Save Text (Utils)",
    "LoadImageList": "Load Image List (Utils)",
    "ImageToList": "Image to List (Utils)",
    "Bypass": "Bypass (Utils)",
    "EmptyLatentRatio": "Empty Latent (Ratio)",
}
