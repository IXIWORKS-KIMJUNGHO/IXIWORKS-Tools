"""
StoryBoard Nodes for ComfyUI
JSON parsing and prompt building for storyboard workflows
"""

import os
import logging
import json

# ComfyUI imports
import folder_paths

logger = logging.getLogger(__name__)

# Register prompt folder for JSON files
PROMPT_FOLDER = os.path.join(folder_paths.get_input_directory(), "prompt")
os.makedirs(PROMPT_FOLDER, exist_ok=True)
folder_paths.folder_names_and_paths["storyboard_prompts"] = (
    [PROMPT_FOLDER],
    {".json"},
)


class JsonParserNode:
    @classmethod
    def INPUT_TYPES(s):
        files = folder_paths.get_filename_list("storyboard_prompts")
        return {
            "required": {
                "JSON": (sorted(files) if files else ["(no files)"],),
            },
        }

    RETURN_TYPES = ("ZIPPED_PROMPT", "ZIPPED_PROMPT", "INT")
    RETURN_NAMES = ("zipped_prompt", "zipped_character", "count")
    FUNCTION = "parse_text"
    CATEGORY = "StoryBoard"
    OUTPUT_IS_LIST = (True, True, False)

    @classmethod
    def IS_CHANGED(s, JSON):
        # Refresh when file changes
        file_path = folder_paths.get_full_path("storyboard_prompts", JSON)
        if file_path and os.path.exists(file_path):
            return os.path.getmtime(file_path)
        return float("nan")

    def parse_text(self, JSON):
        # Use registered folder path
        file_path = folder_paths.get_full_path("storyboard_prompts", JSON)
        if not file_path or not os.path.exists(file_path):
            logger.error(f"[StoryBoard] JsonParserNode: File not found '{json_file}'")
            return ([("", "", "", "")], [("", "", "", "", "", "")], 0)

        logger.info(f"[StoryBoard] JsonParserNode: file path '{file_path}'")
        try:
            # Load the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            scene_data = data.get('scene', {})

            # Process the JSON data
            result = []
            character_result = []

            # New format (camelCase) - extract English fields only
            for item in scene_data.values():
                if isinstance(item, dict):
                    # Extract scene information (English only)
                    time = item.get("time", {}).get("en", "")
                    weather = item.get("weather", {}).get("en", "")
                    camera_shot = item.get("cameraShot", {}).get("en", "")
                    camera_angle = item.get("cameraAngle", {}).get("en", "")
                    description = item.get("description", {}).get("en", "")
                    composition = item.get("composition", {}).get("en", "")

                    # Combine camera shot and angle
                    camera_info = f"{camera_shot}, {camera_angle}" if camera_angle else camera_shot

                    # Combine time and weather
                    time_weather = f"{time}, {weather}" if time and weather else f"{time}{weather}"

                    result.append((description, time_weather, camera_info, composition))

                    # Extract character information for each scene (camelCase)
                    m_char = item.get("mainCharacter", {})
                    s_char = item.get("subCharacter", {})

                    main_char_ko_name = m_char.get("koName", "")
                    main_char_en_name = m_char.get("enName", "")
                    main_char_desc = m_char.get("description", "")

                    sub_char_ko_name = s_char.get("koName", "")
                    sub_char_en_name = s_char.get("enName", "")
                    sub_char_desc = s_char.get("description", "")

                    character_result.append((main_char_ko_name, main_char_en_name, sub_char_ko_name, sub_char_en_name, main_char_desc, sub_char_desc))

            return (result, character_result, len(result))
        except Exception as e:
            logger.error(f"[StoryBoard] JsonParserNode: Error reading file: {e}")
            return ([("", "", "", "")], [("", "", "", "", "", "")], 0)


class BuildCharacterPromptNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"zipped_character": ("ZIPPED_PROMPT",),}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("character_prompt",)
    FUNCTION = "build_character_prompt"
    CATEGORY = "StoryBoard"
    OUTPUT_IS_LIST = (False,)

    def build_character_prompt(self, zipped_character):
        character_prompts = []

        # ComfyUI passes each tuple individually when OUTPUT_IS_LIST is True
        # So zipped_character is a single tuple, not a list of tuples
        if isinstance(zipped_character, tuple) and len(zipped_character) >= 6 and isinstance(zipped_character[0], str):
            # Single character data tuple
            char_data = zipped_character
            main_char_en_name = char_data[1]
            sub_char_en_name = char_data[3]
            main_char_desc = char_data[4]
            sub_char_desc = char_data[5]

            # Build natural language character descriptions
            char_descriptions = []

            if main_char_en_name and main_char_desc:
                desc = main_char_desc.lower()
                if desc.startswith("a ") or desc.startswith("an "):
                    char_descriptions.append(f"{main_char_en_name} is {desc}")
                elif desc.startswith("female") or desc.startswith("male"):
                    char_descriptions.append(f"{main_char_en_name} is a {desc}")
                else:
                    char_descriptions.append(f"{main_char_en_name} is {desc}")

            if sub_char_en_name and sub_char_desc:
                desc = sub_char_desc.lower()
                if desc.startswith("a ") or desc.startswith("an "):
                    char_descriptions.append(f"{sub_char_en_name} is {desc}")
                elif desc.startswith("humanoid") or desc.startswith("robot"):
                    char_descriptions.append(f"{sub_char_en_name} is a {desc}")
                else:
                    char_descriptions.append(f"{sub_char_en_name} is {desc}")

            character_prompt = ". ".join(char_descriptions) + "." if char_descriptions else ""
            logger.info(f"[StoryBoard] BuildCharacterPromptNode: Built prompt: {character_prompt}")
            return (character_prompt,)

        # Fallback: empty result
        logger.warning(f"[StoryBoard] BuildCharacterPromptNode: Invalid input format")
        return ("",)


class BuildPromptNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"zipped_prompt": ("ZIPPED_PROMPT",),}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build_prompt"
    CATEGORY = "StoryBoard"
    OUTPUT_IS_LIST = (True,)

    def build_prompt(self, zipped_prompt):
        prompts = []
        logger.info(f"[StoryBoard] BuildPromptNode: Processing {len(zipped_prompt)} scenes")

        # Handle both list of tuples and single tuple
        if isinstance(zipped_prompt, tuple) and len(zipped_prompt) == 4 and all(isinstance(x, str) for x in zipped_prompt):
            # Single tuple case (from SelectIndexNode)
            zipped_prompt = [zipped_prompt]

        for idx, item in enumerate(zipped_prompt):
            if isinstance(item, tuple) and len(item) >= 4:
                description = item[0]
                time_weather = item[1]
                camera_info = item[2]
                composition = item[3]

                # Build the prompt: camera/lighting first for stronger influence,
                # description last (trigger_words and character are added externally)
                parts = [p for p in [time_weather, camera_info, composition, description] if p.strip()]
                combined_prompt = ", ".join(parts)

                prompts.append(combined_prompt)
                logger.info(f"[StoryBoard] BuildPromptNode: Built prompt for scene {idx+1}")

        logger.info(f"[StoryBoard] BuildPromptNode: Generated {len(prompts)} prompts")
        return (prompts,)


class SelectIndexNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "zipped_prompt": ("ZIPPED_PROMPT",),
                "index": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("ZIPPED_PROMPT",)
    RETURN_NAMES = ("selected_prompt",)
    FUNCTION = "select_index"
    CATEGORY = "StoryBoard"
    OUTPUT_IS_LIST = (False,)

    def select_index(self, zipped_prompt, index):
        logger.info(f"[StoryBoard] SelectIndexNode: Selecting index {index} from {len(zipped_prompt)} items")

        if index < 0 or index >= len(zipped_prompt):
            logger.error(f"[StoryBoard] SelectIndexNode: Index {index} out of range (0-{len(zipped_prompt)-1})")
            # Return empty tuple if index is out of range
            return (("", "", "", ""),)

        selected = zipped_prompt[index]
        logger.info(f"[StoryBoard] SelectIndexNode: Selected item at index {index}")

        return (selected,)


class MergeStringsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strings_a": ("STRING",),
                "strings_b": ("STRING",),
            },
            "optional": {
                "separator": ("STRING", {"default": " "}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_strings",)
    FUNCTION = "merge_strings"
    CATEGORY = "StoryBoard"
    OUTPUT_IS_LIST = (True,)

    def merge_strings(self, strings_a, strings_b, separator=" "):
        # Ensure inputs are lists
        if not isinstance(strings_a, list):
            strings_a = [strings_a]
        if not isinstance(strings_b, list):
            strings_b = [strings_b]

        merged_strings = []

        # Check if arrays have same length
        if len(strings_a) != len(strings_b):
            logger.warning(f"[StoryBoard] MergeStringsNode: Array lengths don't match. strings_a: {len(strings_a)}, strings_b: {len(strings_b)}")
            # Use the shorter length
            min_length = min(len(strings_a), len(strings_b))
        else:
            min_length = len(strings_a)

        logger.info(f"[StoryBoard] MergeStringsNode: Merging {min_length} string pairs")

        for i in range(min_length):
            # Merge with separator
            merged = f"{strings_a[i]}{separator}{strings_b[i]}"

            # Clean up multiple spaces if separator is space
            if separator == " ":
                merged = " ".join(merged.split())

            merged_strings.append(merged)

        return (merged_strings,)


LORA_STYLE_PRESETS = {
    "inksketch": {
        "description": "Monochrome ink sketch style",
        "conflicts": ["vivid colors", "bright colors", "golden hour", "warm glow", "colorful", "orange-gold", "vibrant"],
    },
    "ink-wash": {
        "description": "Traditional Asian ink wash painting style",
        "conflicts": ["vivid colors", "bright colors", "golden hour", "warm glow", "neon", "vibrant"],
    },
    "pen-ink-illustration": {
        "description": "Detailed pen and ink linework illustration",
        "conflicts": ["vivid colors", "bright colors", "golden hour", "warm glow", "colorful", "soft blur"],
    },
    "watercolor-illustration": {
        "description": "Soft watercolor style with gentle color bleeding",
        "conflicts": ["harsh lighting", "neon", "sharp edges", "hyper-realistic", "photorealistic"],
    },
    "ink-watercolor": {
        "description": "Soft ink and watercolor blend with muted tones",
        "conflicts": ["vivid", "harsh", "neon", "bright saturated", "hyper-realistic"],
    },
    "anime": {
        "description": "Japanese anime/manga style",
        "conflicts": ["photorealistic", "realistic photo", "detailed skin texture", "pores", "hyper-realistic", "raw photo"],
    },
    "realistic": {
        "description": "Photorealistic style",
        "conflicts": ["anime", "cartoon", "illustration", "cel-shaded", "flat colors", "lineart"],
    },
    "3d-render": {
        "description": "3D rendered CGI style",
        "conflicts": ["2d", "flat", "hand-drawn", "sketch", "watercolor", "oil painting"],
    },
}


class PromptStyleFilterNode:
    """Filter prompt to remove conflicting keywords for specific LoRA styles using Claude API."""

    SEPARATOR = "\n---PROMPT_SEPARATOR---\n"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "lora_style": (list(LORA_STYLE_PRESETS.keys()),),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_prompt",)
    FUNCTION = "filter_prompt"
    CATEGORY = "StoryBoard"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def filter_prompt(self, prompt, lora_style, api_key):
        # Handle list inputs
        style = lora_style[0] if isinstance(lora_style, list) else lora_style
        key = api_key[0] if isinstance(api_key, list) else api_key
        prompts = prompt if isinstance(prompt, list) else [prompt]

        if not key:
            logger.warning("[StoryBoard] PromptStyleFilterNode: No API key provided, returning original prompts")
            return (prompts,)

        style_info = LORA_STYLE_PRESETS.get(style, {})
        style_desc = style_info.get("description", style)
        conflicts = style_info.get("conflicts", [])

        # Combine all prompts with separator
        combined_input = self.SEPARATOR.join([f"[{i+1}] {p}" for i, p in enumerate(prompts)])

        system_prompt = f"""You are a prompt optimizer for image generation.
Your task is to filter prompts to make them compatible with a specific LoRA style.

Target LoRA style: {style} ({style_desc})

Known conflicting keywords/phrases for this style:
{', '.join(conflicts)}

Instructions:
1. Remove or replace any keywords that conflict with the target style
2. Keep the core subject and composition intact
3. Do not add new creative elements - only remove conflicts
4. You will receive multiple numbered prompts separated by "---PROMPT_SEPARATOR---"
5. Return each filtered prompt on the same format: [number] filtered_prompt
6. Use the EXACT same separator "---PROMPT_SEPARATOR---" between outputs
7. If no conflicts found in a prompt, return it unchanged"""

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)

            response = client.messages.create(
                model="claude-haiku-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Filter these prompts:\n\n{combined_input}"}
                ]
            )

            result_text = response.content[0].text.strip()

            # Split and parse results
            filtered_results = []
            parts = result_text.split("---PROMPT_SEPARATOR---")

            for part in parts:
                part = part.strip()
                # Remove [number] prefix if present
                if part.startswith("["):
                    idx = part.find("]")
                    if idx != -1:
                        part = part[idx+1:].strip()
                if part:
                    filtered_results.append(part)

            # Ensure we have the same number of outputs
            if len(filtered_results) != len(prompts):
                logger.warning(f"[StoryBoard] PromptStyleFilterNode: Output count mismatch ({len(filtered_results)} vs {len(prompts)}), returning originals")
                return (prompts,)

            logger.info(f"[StoryBoard] PromptStyleFilterNode: Filtered {len(prompts)} prompts for {style}")
            return (filtered_results,)

        except ImportError:
            logger.error("[StoryBoard] PromptStyleFilterNode: anthropic package not installed")
            return (prompts,)
        except Exception as e:
            logger.error(f"[StoryBoard] PromptStyleFilterNode: API error - {e}")
            return (prompts,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "JsonParserNode": JsonParserNode,
    "BuildPromptNode": BuildPromptNode,
    "BuildCharacterPromptNode": BuildCharacterPromptNode,
    "SelectIndexNode": SelectIndexNode,
    "MergeStringsNode": MergeStringsNode,
    "PromptStyleFilter": PromptStyleFilterNode,
}

# Display name mappings for ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "JsonParserNode": "JSON Parser (StoryBoard)",
    "BuildPromptNode": "Build Prompt (StoryBoard)",
    "BuildCharacterPromptNode": "Build Character Prompt (StoryBoard)",
    "SelectIndexNode": "Select Index (StoryBoard)",
    "MergeStringsNode": "Merge Strings (StoryBoard)",
    "PromptStyleFilter": "Prompt Style Filter (StoryBoard)",
}
