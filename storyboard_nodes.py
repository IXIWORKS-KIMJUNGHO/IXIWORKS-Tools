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
        "color_mode": "inksketch",  # specific handling for ink sketch
        "add_keywords": "monochrome, high contrast linework, detailed crosshatching, sketch texture",
    },
    "ink-wash": {
        "name": "Ink Wash",
        "color_mode": "inkwash",  # specific handling for ink wash
        "add_keywords": "atmospheric tonal depth, grayscale",
    },
    "pen-ink-illustration": {
        "name": "Pen & Ink Illustration",
        "color_mode": "penink",  # specific handling
        "add_keywords": "black and white, high contrast ink-illustration",
    },
    "ink-watercolor": {
        "name": "Ink and Watercolor",
        "color_mode": "inkwatercolor",  # specific handling
        "add_keywords": "warm earthy tones, soft watercolor background",
    },
    "watercolor-illustration": {
        "name": "Watercolor Illustration",
        "color_mode": "watercolorillust",  # specific handling
        "add_keywords": "vibrant warm palette, gentle wash effects with soft color bleeding",
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

        # Get style-specific keywords to add
        add_keywords = style_info.get("add_keywords", "")

        # Combine all prompts with separator
        combined_input = self.SEPARATOR.join([f"[{i+1}] {p}" for i, p in enumerate(prompts)])

        # Common instruction for all styles
        remove_realistic = """
## REMOVE photorealistic keywords (CRITICAL):
- photo realistic, photorealistic, realistic, hyper realistic, hyperrealistic
- photograph, photo, photography, RAW photo
- DSLR, 8k uhd, 4k, ultra realistic, lifelike
- cinematic photo, film grain (unless style-appropriate)
These keywords override LoRA styles and MUST be removed."""

        keep_instruction = """
## ALWAYS KEEP (never modify these):
- Composition: rule of thirds, leading lines, centered, off-center, etc.
- Camera angle: wide shot, medium shot, close-up, eye-level, low angle, high angle, etc.
- Camera position descriptions
- Subject pose and position descriptions
- Framing descriptions"""

        if color_mode == "inksketch":
            system_prompt = f"""You are a prompt optimizer for ink sketch style image generation.
{remove_realistic}

## Transform these elements for ink sketch style:

### Colors → REMOVE ALL:
- Remove all color adjectives: beige, olive, white, red, blue, golden, etc.
- "beige hoodie" → "hoodie", "red car" → "car"

### Lighting → Convert to contrast-based:
- "golden hour" → "low angle dramatic lighting"
- "warm glow" → "strong directional lighting"
- "bathed in sunlight" → "harsh angular shadows"
- Keep: shadows, contrast, silhouette
{keep_instruction}

## Output Format:
- Multiple prompts separated by "---PROMPT_SEPARATOR---"
- Format: [1] filtered_prompt
- Place style-defining keywords at the START of each prompt"""

        elif color_mode == "inkwash":
            system_prompt = f"""You are a prompt optimizer for ink wash style image generation.
{remove_realistic}

## Transform these elements for ink wash style:

### Colors:
- KEEP muted colors: beige, olive, gray, brown
- REMOVE vivid colors: neon, vibrant, bright saturated
- REMOVE pure white/black adjectives from objects

### Lighting → Convert to atmospheric:
- "bright sunlight" → "morning light casting long diagonal shadows"
- Add: "wet pavement", "puddle reflections"

### Special: CONDENSE the prompt
- Remove verbose explanations, keep core descriptions
{keep_instruction}

## Output Format:
- Multiple prompts separated by "---PROMPT_SEPARATOR---"
- Format: [1] filtered_prompt
- Place style-defining keywords at the START of each prompt"""

        elif color_mode == "penink":
            system_prompt = f"""You are a prompt optimizer for pen and ink illustration style image generation.
{remove_realistic}

## Transform these elements for pen & ink style:

### Colors:
- KEEP muted colors: beige, olive, white, gray, brown
- REMOVE vivid/neon colors

### Lighting → Convert with crosshatching:
- "bathed in glow" → "low angle afternoon light casting long dramatic shadows"
- "well-defined shadows" → "well-defined shadows with crosshatching"

### Technique → ENHANCE details:
- "brick buildings" → "detailed brick buildings"
- "cafe sign" → "hanging cafe sign"
- "city street" → "cobblestone city street"
- Add: "intricate linework on architectural details"
{keep_instruction}

## Output Format:
- Multiple prompts separated by "---PROMPT_SEPARATOR---"
- Format: [1] filtered_prompt
- Place style-defining keywords at the START of each prompt"""

        elif color_mode == "inkwatercolor":
            system_prompt = f"""You are a prompt optimizer for ink and watercolor illustration style image generation.
{remove_realistic}

## Transform these elements for ink watercolor style:

### Colors:
- KEEP ALL warm colors: golden hour, warm glow, warm sunlight, amber
- KEEP muted colors: beige, olive, brown, earthy
- REMOVE only: neon, cold harsh colors

### Lighting → Keep warm, enhance atmosphere:
- Keep all warm lighting descriptions
- Add: "loose ink linework with watercolor washes"

### Technique → ENHANCE with charm:
- "city street" → "dusty city street"
- "brick buildings" → "charming brick buildings"
- "cafe sign" → "hand-painted cafe sign"
- Consider adding: "a stray dog nearby", "pigeons"
{keep_instruction}

## Output Format:
- Multiple prompts separated by "---PROMPT_SEPARATOR---"
- Format: [1] filtered_prompt
- Place style-defining keywords at the START of each prompt"""

        elif color_mode == "watercolorillust":
            system_prompt = f"""You are a prompt optimizer for watercolor illustration style image generation.
{remove_realistic}

## Transform these elements for watercolor illustration style:

### Colors → KEEP ALL:
- Keep all colors including vibrant (vibrant, colorful, golden, warm)

### Lighting → Keep and enhance:
- Keep all warm/natural lighting
- Add: "dappled light filtering through plane trees"

### Technique → ENHANCE with color:
- "city street" → "sunlit cobblestone city street"
- "brick buildings" → "colorful brick buildings"
- "cafe sign" → "charming cafe sign with striped awning"
- Add: "wet-on-wet blending with soft edges"
{keep_instruction}

## Output Format:
- Multiple prompts separated by "---PROMPT_SEPARATOR---"
- Format: [1] filtered_prompt
- Place style-defining keywords at the START of each prompt"""

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)

            logger.info(f"[StoryBoard] PromptStyleFilterNode: Filtering {len(prompts)} prompts with style '{style_name}' (color_mode: {color_mode})")

            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Filter these prompts:\n\n{combined_input}"}
                ]
            )

            result_text = response.content[0].text.strip()
            logger.info(f"[StoryBoard] PromptStyleFilterNode: Received response from Claude API")

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

            # Prepend add_keywords to each result for stronger LoRA trigger
            if add_keywords:
                filtered_results = [f"{add_keywords}, {r}" for r in filtered_results]
                logger.info(f"[StoryBoard] PromptStyleFilterNode: Prepended keywords: '{add_keywords}'")

            # Log each filtered result (truncated for readability)
            for i, (orig, filtered) in enumerate(zip(prompts, filtered_results)):
                orig_preview = orig[:50] + "..." if len(orig) > 50 else orig
                filtered_preview = filtered[:80] + "..." if len(filtered) > 80 else filtered
                logger.info(f"[StoryBoard] PromptStyleFilterNode: [{i+1}] {orig_preview} → {filtered_preview}")

            logger.info(f"[StoryBoard] PromptStyleFilterNode: Done - {len(filtered_results)} prompts filtered for '{style_name}'")
            return (filtered_results,)

        except ImportError:
            logger.error("[StoryBoard] PromptStyleFilterNode: anthropic package not installed")
            return (prompts,)
        except Exception as e:
            logger.error(f"[StoryBoard] PromptStyleFilterNode: API error - {e}")
            return (prompts,)


class PromptPoseFilterNode:
    """Filter pose-related keywords from prompt when using Pose ControlNet."""

    SEPARATOR = "\n---PROMPT_SEPARATOR---\n"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_prompt",)
    FUNCTION = "filter_prompt"
    CATEGORY = "StoryBoard"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def filter_prompt(self, prompt, api_key):
        # Handle list inputs
        key = api_key[0] if isinstance(api_key, list) else api_key
        prompts = prompt if isinstance(prompt, list) else [prompt]

        if not key:
            logger.warning("[StoryBoard] PromptPoseFilterNode: No API key provided, returning original prompts")
            return (prompts,)

        # Combine all prompts with separator
        combined_input = self.SEPARATOR.join([f"[{i+1}] {p}" for i, p in enumerate(prompts)])

        system_prompt = """You are a prompt optimizer that removes pose-related keywords for Pose ControlNet usage.

## REMOVE these pose/posture keywords:
- Body postures: standing, sitting, kneeling, lying down, crouching, leaning, bending, squatting
- Actions/movements: walking, running, jumping, dancing, waving, reaching, stretching
- Arm positions: arms crossed, arms raised, arms akimbo, hands on hips, pointing, holding, arms behind back
- Hand gestures: peace sign, thumbs up, fist, open palm, clasped hands
- Leg positions: legs crossed, one leg raised, spread legs, kicking
- Head/gaze direction: looking at viewer, looking away, looking up, looking down, looking left/right, head tilted, turned head
- Body orientation: from behind, from side, profile view, back view, three-quarter view
- Full body descriptors: full body pose, dynamic pose, action pose, relaxed pose, tense pose

## KEEP these (do NOT remove):
- Camera angles: low angle, high angle, eye level, bird's eye view, worm's eye view
- Camera shots: wide shot, medium shot, close-up, extreme close-up, full shot
- Composition: rule of thirds, centered, off-center, leading lines
- Facial expressions: smiling, serious, surprised, angry, sad (expressions are NOT poses)
- Scene descriptions, lighting, colors, clothing, background

## Output Format:
- Multiple prompts separated by "---PROMPT_SEPARATOR---"
- Format: [1] filtered_prompt
- Keep the prompt natural and coherent after removing pose keywords"""

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)

            logger.info(f"[StoryBoard] PromptPoseFilterNode: Filtering {len(prompts)} prompts for pose keywords")

            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Remove pose keywords from these prompts:\n\n{combined_input}"}
                ]
            )

            result_text = response.content[0].text.strip()
            logger.info(f"[StoryBoard] PromptPoseFilterNode: Received response from Claude API")

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
                logger.warning(f"[StoryBoard] PromptPoseFilterNode: Output count mismatch ({len(filtered_results)} vs {len(prompts)}), returning originals")
                return (prompts,)

            # Log each filtered result (truncated for readability)
            for i, (orig, filtered) in enumerate(zip(prompts, filtered_results)):
                orig_preview = orig[:50] + "..." if len(orig) > 50 else orig
                filtered_preview = filtered[:80] + "..." if len(filtered) > 80 else filtered
                logger.info(f"[StoryBoard] PromptPoseFilterNode: [{i+1}] {orig_preview} → {filtered_preview}")

            logger.info(f"[StoryBoard] PromptPoseFilterNode: Done - {len(filtered_results)} prompts filtered")
            return (filtered_results,)

        except ImportError:
            logger.error("[StoryBoard] PromptPoseFilterNode: anthropic package not installed")
            return (prompts,)
        except Exception as e:
            logger.error(f"[StoryBoard] PromptPoseFilterNode: API error - {e}")
            return (prompts,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "JsonParserNode": JsonParserNode,
    "BuildPromptNode": BuildPromptNode,
    "BuildCharacterPromptNode": BuildCharacterPromptNode,
    "SelectIndexNode": SelectIndexNode,
    "MergeStringsNode": MergeStringsNode,
    "PromptStyleFilter": PromptStyleFilterNode,
    "PromptPoseFilter": PromptPoseFilterNode,
}

# Display name mappings for ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "JsonParserNode": "JSON Parser (StoryBoard)",
    "BuildPromptNode": "Build Prompt (StoryBoard)",
    "BuildCharacterPromptNode": "Build Character Prompt (StoryBoard)",
    "SelectIndexNode": "Select Index (StoryBoard)",
    "MergeStringsNode": "Merge Strings (StoryBoard)",
    "PromptStyleFilter": "Prompt Style Filter (StoryBoard)",
    "PromptPoseFilter": "Prompt Pose Filter (StoryBoard)",
}
