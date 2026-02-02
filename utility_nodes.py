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
                "on_false": (ANY,),
                "on_true": (ANY,),
                "boolean_switch": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    CATEGORY = "IXIWORKS/Utils"

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
                "string_a": ("STRING", {"default": "", "multiline": True}),
                "string_b": ("STRING", {"default": "", "multiline": True}),
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
            f"input_{i}": (ANY,)
            for i in range(cls.MAX_INPUTS)
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    CATEGORY = "IXIWORKS/Utils"

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
        required = {
            "count": ("INT", {"default": 4, "min": 1, "max": cls.MAX_INPUTS, "step": 1}),
        }
        optional = {
            f"image_{i}": ("IMAGE",)
            for i in range(1, cls.MAX_INPUTS + 1)
        }
        return {"required": required, "optional": optional}

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
    "JoinStrings": JoinStringsNode,
    "SwitchCase": SwitchCaseNode,
    "SaveText": SaveTextNode,
    "LoadImageList": LoadImageListNode,
    "ImageToList": ImageToListNode,
    "Bypass": BypassNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwitchBoolean": "Switch (Utils)",
    "StringToList": "String to List (Utils)",
    "JoinStrings": "Join Strings (Utils)",
    "SwitchCase": "Switch Case (Utils)",
    "SaveText": "Save Text (Utils)",
    "LoadImageList": "Load Image List (Utils)",
    "ImageToList": "Image to List (Utils)",
    "Bypass": "Bypass (Utils)",
}
