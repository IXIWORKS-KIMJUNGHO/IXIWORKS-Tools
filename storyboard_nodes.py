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
        "name": "Ink Sketch",
        "color_mode": "monochrome",  # black and white only
    },
    "ink-wash": {
        "name": "Ink Wash",
        "color_mode": "monochrome",  # grayscale only
    },
    "pen-ink-illustration": {
        "name": "Pen & Ink Illustration",
        "color_mode": "monochrome",  # black and white, sepia
    },
    "ink-watercolor": {
        "name": "Ink and Watercolor",
        "color_mode": "warm-muted",  # warm sepia, earthy, muted tones only
    },
    "watercolor-illustration": {
        "name": "Watercolor Illustration",
        "color_mode": "watercolor",  # soft, natural colors - no harsh/digital
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
        style_name = style_info.get("name", style)
        color_mode = style_info.get("color_mode", "full-color")

        # Full color mode: no filtering needed
        if color_mode == "full-color":
            return (prompts,)

        # Combine all prompts with separator
        combined_input = self.SEPARATOR.join([f"[{i+1}] {p}" for i, p in enumerate(prompts)])

        if color_mode == "monochrome":
            system_prompt = f"""You are a prompt filter for monochrome image generation.

## Task
Remove ALL color-related words and phrases from the prompt. The target style "{style_name}" is MONOCHROME (black and white / grayscale only).

## What to REMOVE:
- All color names (red, blue, green, golden, orange, pink, purple, neon, etc.)
- Colored lighting (neon lights, neon glow, warm light, cool light, golden hour, colored illumination)
- Color descriptions (vivid colors, bright colors, colorful, vibrant, saturated)
- Colored reflections (shimmering colored pools, rainbow reflections)
- Color temperature terms (warm tones, cool tones, sepia)

## What to KEEP (do not modify):
- Subject descriptions (person, clothing, pose, expression)
- Composition (camera angle, shot type, framing)
- Lighting intensity (bright, dim, harsh, soft - without color)
- Atmosphere (dark, moody, shadows, contrast, silhouette, atmospheric)
- Location, setting, textures, materials

## Output Format:
- Multiple prompts separated by "---PROMPT_SEPARATOR---"
- Format: [1] filtered_prompt---PROMPT_SEPARATOR---[2] filtered_prompt
- Return ONLY the filtered prompts, no explanations"""

        elif color_mode == "warm-muted":
            system_prompt = f"""You are a prompt filter for warm-toned ink and watercolor style image generation.

## Task
Remove harsh/vivid colors but KEEP warm, earthy, muted tones. The target style "{style_name}" uses warm sepia, earthy, and muted watercolor tones.

## What to REMOVE:
- Neon colors and lighting (neon lights, neon glow, neon-lit)
- Vivid/saturated colors (vivid, vibrant, saturated, bright colors)
- Cool tones (cool light, blue tones, cyan, cold)
- Modern artificial lighting colors
- Rainbow, iridescent effects

## What to KEEP (these are compatible):
- Warm tones (warm, golden, amber, sepia, earthy)
- Muted colors (muted, soft, gentle, faded)
- Natural warm light (afternoon light, sunset warmth, warm sunlight)
- Earthy colors (brown, ochre, terracotta, dusty)
- Watercolor descriptors (soft wash, blending, tones)

## What to KEEP (do not modify):
- Subject descriptions (person, clothing, pose, expression)
- Composition (camera angle, shot type, framing)
- Atmosphere and mood
- Location, setting, textures, materials

## Output Format:
- Multiple prompts separated by "---PROMPT_SEPARATOR---"
- Format: [1] filtered_prompt---PROMPT_SEPARATOR---[2] filtered_prompt
- Return ONLY the filtered prompts, no explanations"""

        elif color_mode == "watercolor":
            system_prompt = f"""You are a prompt filter for watercolor illustration style image generation.

## Task
Remove harsh/digital/photorealistic elements. The target style "{style_name}" uses soft, natural watercolor aesthetics with gentle color blending.

## What to REMOVE:
- Neon lights and artificial colored lighting (neon, neon-lit, LED)
- Photorealistic/digital terms (photorealistic, hyper-realistic, raw photo, 8k, HDR)
- Harsh contrast terms (harsh shadows, sharp contrast, hard edges)
- Modern digital effects (lens flare, bokeh, chromatic aberration)

## What to KEEP (these are compatible):
- Soft, natural colors (any natural color is fine)
- Warm lighting (golden hour, afternoon light, sunset, dappled light)
- Watercolor descriptors (soft, wet, blending, wash, gentle, diffused)
- Vibrant but natural colors (vibrant, colorful - these are OK)
- Organic textures and natural materials

## What to KEEP (do not modify):
- Subject descriptions (person, clothing, pose, expression)
- Composition (camera angle, shot type, framing)
- Atmosphere and mood
- Location, setting, textures, materials

## Output Format:
- Multiple prompts separated by "---PROMPT_SEPARATOR---"
- Format: [1] filtered_prompt---PROMPT_SEPARATOR---[2] filtered_prompt
- Return ONLY the filtered prompts, no explanations"""

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)

            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
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
