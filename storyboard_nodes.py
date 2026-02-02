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


class JsonParserNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"file_name": ("STRING", {"default": "prompt.json"})},
        }

    RETURN_TYPES = ("ZIPPED_PROMPT", "ZIPPED_PROMPT", "INT")
    RETURN_NAMES = ("zipped_prompt", "zipped_character", "count")
    FUNCTION = "parse_text"
    CATEGORY = "StoryBoard"
    OUTPUT_IS_LIST = (True, True, False)

    def parse_text(self, file_name):
        # Use ComfyUI's input directory
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, "prompt", file_name)
        file_path = os.path.abspath(os.path.normpath(file_path))

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


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "JsonParserNode": JsonParserNode,
    "BuildPromptNode": BuildPromptNode,
    "BuildCharacterPromptNode": BuildCharacterPromptNode,
    "SelectIndexNode": SelectIndexNode,
    "MergeStringsNode": MergeStringsNode,
}

# Display name mappings for ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "JsonParserNode": "JSON Parser (StoryBoard)",
    "BuildPromptNode": "Build Prompt (StoryBoard)",
    "BuildCharacterPromptNode": "Build Character Prompt (StoryBoard)",
    "SelectIndexNode": "Select Index (StoryBoard)",
    "MergeStringsNode": "Merge Strings (StoryBoard)",
}
