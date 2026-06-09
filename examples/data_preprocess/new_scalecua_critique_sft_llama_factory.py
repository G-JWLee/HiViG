# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
from io import BytesIO

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

CRITIC_GENERATION_FORMAT = """
You are an expert GUI Environment Simulator and Action Critic for a GUI navigation agent.

### Task Description
# You will be provided with:
- User Instruction (Goal): The overall, high-level task the user wants the agent to accomplish.
- Trajectory: A sequential record of the agent's past proposed steps.
- Current Observation: The screenshot visual representation of the current screen. **Note: IF the agent's proposed step involves spatial coordinates, a visual marker (red 'X' or blue arrow) has been injected into this image. If the action is non-spatial (e.g., type, wait), the image is clean.**
- Current Proposed Step: The combined verbalized intent (Action: ...) and the executable command (<tool_call>...</tool_call>) suggested by the agent for the current observation.

Your task is to predict the physical and logical consequences of the agent's action and provide goal-oriented environmental feedback so the agent can deduce the correct action itself.

### Action Space
{platform_action_space}

### Error Dimension Taxonomy
{error_dimension_str}

### Output Format
You must strictly follow this exact structure. Before predicting the final evaluation, you must reason through the visual state and logical deduction step-by-step inside a <think> block.

<think>
In your monologue, you must naturally flow through these logical steps:

In your first paragraph, start exactly with this format: "Action String Check: [Insert the EXACT raw <tool_call> string here] - Syntax is [Valid/Invalid]." (Note any JSON parsing errors if invalid).
**CRITICAL VISUAL GROUNDING:** Next, you MUST evaluate ALL `<tool_call>` blocks in the 'Current Proposed Step' based on the arguments provided in the JSON.
- **If ANY tool call contains `coordinate` arguments (e.g., click, swipe, mouse_move):** The coordinates perfectly match the injected visual marker. Locate the **red 'X' marker** or the **blue arrow**. To prevent spatial hallucinations, you must perform a strict, literal visual scan:
    1. **Literal Center:** First, read the exact text or identify the specific icon located directly beneath the exact center intersection of the marker.
    2. **Adjacency Check:** Identify the elements immediately adjacent (left/right/above/below) to the marker.
    3. **Final Identification:** Based strictly on the literal center, explicitly describe exactly what UI element (or empty space) is targeted. (e.g., "The X is directly on the 'Subscribe' text, adjacent to the '0' counter. Therefore, it targets the Subscribe button, not the counter.")
- **If NO tool calls in the 'Current Proposed Step' contain `coordinate` arguments (e.g., ONLY scroll, type, key, wait, terminate):** You are STRICTLY FORBIDDEN from looking for or mentioning a red 'X' or blue arrow. **CRITICAL WARNING: If you see a red 'X' in the image during a non-coordinate action, it is a residual artifact from a previous step. You MUST completely ignore it.** Evaluate the global screen state and the logical parameters provided in the 'Current Proposed Step' JSON.
Next, analyze the 'Trajectory' against the 'User Instruction (Goal)' to establish the current progress state.
Finally, if a verbal intent (Action: ...) is provided, explicitly compare what the agent claims it is doing against the visual reality you just observed in the context of this current state, calling out any contradictions. If no verbal intent is provided, explicitly state what UI element the action targets in the context of the current state.

In your second paragraph, using your knowledge of GUI interfaces, predict exactly what will physically happen to the UI layout if this specific action is executed. Use forward-predictive future tense.

In your third paragraph, compare your predicted consequence directly to the 'User Instruction (Goal)'. You must organically deduce why this predicted structural UI change succeeds or fails. If it succeeds, conclude 'Good'. If it fails, conclude 'Bad', state the matching 'Target Error Dimension', and explain the UI physics rule that the agent violated.
</think>
<criticism>
[If the action is correct, output exactly this format:]
Overall Grading: Good
Explanation: [Describe the immediate physical/structural consequence of this action in the future tense. Then, explicitly state the universal UI rule or environmental mechanic that makes this interaction successful.]

[If the action is flawed, output exactly this format:]
Overall Grading: Bad
Error Dimension: [You MUST select the exact matching category from the Error Dimension Taxonomy above].
Explanation: [Describe the immediate physical/structural consequence of the flawed action in the future tense. Then, state the universal UI rule the agent violated, and explain the correct environmental mechanic required for this type of interaction.]
</criticism>
""".strip()




error_dimensions = [
    "Grounding/Spatial Error: Semantic intent is correct, but coordinate precision fails. Near-misses the target, landing in adjacent dead space (e.g., a minor pixel offset just outside the bounding box).",
    "Procedural Prerequisite Neglect: Skips a mandatory preceding state change. Either fails to execute a preparatory action (e.g., focusing a field before typing), or fails to dismiss a foreground overlay blocking the target.",
    "Semantic Error: Targets perfectly, but misinterprets vocabulary, icons, or UI paradigms (e.g., clicking 'Sign Up' instead of 'Log In', or clicking a deceptive ad).",
    "Termination Misjudgment: Misjudges task completion. Prematurely outputs 'terminate', fails to explicitly report data using 'answer', or hallucinates redundant steps after the goal is met.",
    "Constraint Neglect: Ignores a specific attribute or positional constraint explicitly stated in the goal (e.g., selecting the wrong author or wrong position). If not explicitly requested, it is not neglected.",
    "Action Formulation Error: Intent is correct, but the JSON crashes the parser. Includes syntax errors (e.g., missing quotes, trailing commas, mismatched braces) or missing required arguments/invalid enums.",
    "Observation Neglect: Attempts to search, scroll, or open menus for targets that are already clearly visible on the screen.",
    "Suboptimal Path: Selects highly inefficient micro-actions (e.g., repetitive arrow or backspace clicks) instead of standard faster paradigms (e.g., direct text entry, bulk delete).",
    "Parameter Vector Miscalibration: Fails physical vector execution. Reasons or executes the exact opposite direction of the goal (Polarity Reversal, e.g., wrong mathematical sign for scrolling) or uses a drastically undersized scale resulting in negligible UI movement (Magnitude Insufficiency, e.g., undersized scale resulting in negligible movement).",
    "Visual Hallucination: Interacts with a strictly non-existent UI element. Interacting with a ghost element from a previous state, or blindly guessing a layout coordinate.",
    "Timing and Latency Neglect: Executes prematurely, ignoring system busy indicators (e.g., loading spinners, unfolding menus, disabled buttons).",
    "Action-Operation Misalignment: Verbalized intent drastically contradicts the executable JSON string. The reasoned action description is logical, but the actual generated <tool_call> is completely different or nonsensical.",
]
error_dimension_str = "\n".join([f"- {dim}" for dim in error_dimensions])



action_spaces = {
    "desktop": """
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
* `interact`: Resolve the blocking window by interacting with the user.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click", "scroll", "hscroll", "wait", "terminate", "answer", "interact"], "type": "string"}, "keys": {"description": "Required only by `action=key`.", "type": "array"}, "text": {"description": "Required only by `action=type`, `action=answer` and `action=interact`.", "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll` and `action=hscroll`.", "type": "number"}, "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>
""".strip(),

    "mobile": """
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
  - This supports adb's `keyevent` syntax.
  - Examples: \"volume_up\", \"volume_down\", \"power\", \"camera\", \"clear\".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `answer`: Terminate the current task and output the answer.
* `interact`: Resolve the blocking window by interacting with the user.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "answer", "interact", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=open`, `action=answer`,and `action=interact`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>
""".strip()
}



USER_PROMPT = """
User Instruction (Goal):
{intent}

Trajectory:
{trajectory}

Current Observation:
<image>

Current Proposed Step:
{current_proposed_step}
""".strip()

def load_processor(path: str):
    from transformers import AutoProcessor
    # Use AutoProcessor to handle both vision and language modalities
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=False)
    return processor


def load_image_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()


def process_image(image: dict | Image.Image, image_patch_size: int = 14) -> Image.Image:
    from qwen_vl_utils import fetch_image

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = Image.open(BytesIO(image["bytes"]))

    try:
        ans = fetch_image(image, image_patch_size=image_patch_size)
    except Exception:
        ans = fetch_image(image)
    return ans


def get_or_create_marked_image(image_path: str, action_str: str, output_dir: str) -> str:
    """Draws markers on the image if needed, saves it, and returns the new path."""
    if not os.path.exists(image_path):
        return image_path

    # Generate a unique filename based on the original image and the specific action
    pair_hash = hashlib.md5(f"{image_path}_{action_str}".encode('utf-8')).hexdigest()
    save_path = os.path.join(output_dir, f"marked_{pair_hash}.png")

    # If we already created this exact image in a previous run, just return it
    if os.path.exists(save_path):
        return save_path

    # Find tool calls
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(action_str)

    if not matches:
        return image_path # No action to draw, return original path

    point_actions = {"left_click", "click", "right_click", "double_click", "triple_click", "middle_click", "mouse_move", "long_press"}
    drag_actions = {"swipe", "left_click_drag"}
    shapes_to_draw = []

    for match_text in matches:
        try:
            tool_call = ast.literal_eval(match_text.strip())
            args = tool_call.get("arguments", {})
            action_type = args.get("action")
            
            if action_type in point_actions and "coordinate" in args:
                shapes_to_draw.append(("point", args["coordinate"]))
            elif action_type in drag_actions and "coordinate" in args and "coordinate2" in args:
                shapes_to_draw.append(("arrow", args["coordinate"], args["coordinate2"]))
        except:
            continue

    if not shapes_to_draw:
        return image_path # No spatial actions, return original path

    # Open, draw, and save
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        # Determine environment based on aspect ratio
        is_mobile = height > width
        base_dim = min(width, height)

        if is_mobile:
            # --- MOBILE SCALING (Large Touch Targets) ---
            # Larger gap to preserve the center of large app icons
            gap = max(6, base_dim // 80) 
            # Much wider arms to span the large hit-boxes
            outer = gap + max(12, base_dim // 40)
            # Thicker lines so the signal survives patch compression
            line_thickness = max(4, base_dim // 150) 
            arrow_head_size = max(10, base_dim // 60)
        else:
            # --- DESKTOP SCALING (Dense Layouts) ---
            gap = max(3, base_dim // 150) 
            outer = gap + max(4, base_dim // 120)
            line_thickness = max(2, base_dim // 300) 
            arrow_head_size = max(6, base_dim // 120)

        for shape_type, *coords in shapes_to_draw:
            if shape_type == "point":
                rel_x, rel_y = coords[0][0] / 1000.0, coords[0][1] / 1000.0
                if 0 <= rel_x <= 1 and 0 <= rel_y <= 1:
                    x, y = int(rel_x * width), int(rel_y * height)
                    
                    # Draw "Broken X" (Diagonal Crosshair)
                    # Top-Left arm
                    draw.line([(x - outer, y - outer), (x - gap, y - gap)], fill="red", width=line_thickness)
                    # Bottom-Right arm
                    draw.line([(x + gap, y + gap), (x + outer, y + outer)], fill="red", width=line_thickness)
                    # Top-Right arm
                    draw.line([(x + outer, y - outer), (x + gap, y - gap)], fill="red", width=line_thickness)
                    # Bottom-Left arm
                    draw.line([(x - gap, y + gap), (x - outer, y + outer)], fill="red", width=line_thickness)
            
            elif shape_type == "arrow":
                rel_x1, rel_y1 = coords[0][0] / 1000.0, coords[0][1] / 1000.0
                rel_x2, rel_y2 = coords[1][0] / 1000.0, coords[1][1] / 1000.0
                if (0 <= rel_x1 <= 1 and 0 <= rel_y1 <= 1 and 0 <= rel_x2 <= 1 and 0 <= rel_y2 <= 1):
                    x1, y1 = int(rel_x1 * width), int(rel_y1 * height)
                    x2, y2 = int(rel_x2 * width), int(rel_y2 * height)

                    angle = math.atan2(y2 - y1, x2 - x1)

                    pullback = arrow_head_size * 0.5
                    line_end_x = x2 - pullback * math.cos(angle)
                    line_end_y = y2 - pullback * math.sin(angle)

                    draw.line([(x1, y1), (line_end_x, line_end_y)], fill="blue", width=line_thickness)

                    p1 = (x2 - arrow_head_size * math.cos(angle - math.pi / 6), y2 - arrow_head_size * math.sin(angle - math.pi / 6))
                    p2 = (x2 - arrow_head_size * math.cos(angle + math.pi / 6), y2 - arrow_head_size * math.sin(angle + math.pi / 6))
                    
                    # Draw a solid polygon for the arrowhead pointing exactly at (x2, y2)
                    draw.polygon([(x2, y2), p1, p2], fill="blue")

        img.save(save_path)
        return save_path
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        return image_path


def parse_think(target):
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, target, re.DOTALL)
    thought = think_match.group(1).strip() if think_match else ""
    text = re.sub(think_pattern, "", target, flags=re.DOTALL).strip()

    return thought, text


def fix_rationale_structure_simple(rationale: str) -> str:
    """
    Finds the 'Action String Check' line. If it is immediately followed by 
    two or more newlines (\n\n), it replaces those newlines with a space, 
    merging it seamlessly with the next sentence.
    """
    # Regex: Capture the exact line starting with "Action String Check:" up to its end,
    # and match if it is followed by 2 or more newlines.
    pattern = r"(Action String Check:[^\n]+)\n{2,}"
    
    # Replace the matched line + newlines with the line + a single space.
    # count=1 ensures we only target the very first occurrence.
    return re.sub(pattern, r"\1 ", rationale, count=1)



def process_warmup_rationale(task_sample, split, idx, processor, max_num_token, annotated_dir):
    task = task_sample.get("task")
    trajectory = task_sample.get("trajectory")
    proposed_step = task_sample.get("proposed_step")
    original_image_path = task_sample.get("image_path")

    # NEW: Create the marked image (or get the original path if no marking needed)
    final_image_path = get_or_create_marked_image(original_image_path, proposed_step, annotated_dir)
    current_observation = load_image_bytes(final_image_path)
    # Get absolute path to ensure LLaMA-Factory can always find it
    abs_image_path = os.path.abspath(final_image_path)

    if task_sample['environment'] == 'android':
        platform = 'mobile'
    elif task_sample['environment'] in ['web', 'windows', 'mac']:
        platform = 'desktop'
    else:
        ValueError(f"{task_sample['environment']} is not defined.")

    trajectory_str = ""
    for i, step in enumerate(trajectory):
        trajectory_str += (
            f"Step {i+1}:\n"
            f"Action: {step.get('operation')}\n"
            f"{step.get('action')}\n\n"
        )
    trajectory_str = trajectory_str.strip()

    rationale, target = parse_think(task_sample['target_output'])
    
    if rationale is None or target is None:
        return None

    rationale = fix_rationale_structure_simple(rationale)

    previous_was_answer = False
    if len(trajectory) > 0:
        last_step = str(trajectory[-1].get('action', ''))
        if '"action": "answer"' in last_step:
            previous_was_answer = True

    # 2. Check if current proposed action is 'terminate'
    current_is_terminate = '"action": "terminate"' in proposed_step

    # 3. Check if the rationale flags this as 'Termination Misjudgment'
    is_termination_misjudgment = "Termination Misjudgment" in rationale

    # 4. If all three are true, this is a flawed synthetic sample. Drop it.
    if previous_was_answer and current_is_terminate and is_termination_misjudgment:
        # Optional: print(f"Filtered out false Termination Misjudgment at idx {idx}")
        return None

    system_prompt = CRITIC_GENERATION_FORMAT.format(platform_action_space=action_spaces[platform], error_dimension_str=error_dimension_str)
    user_prompt = USER_PROMPT.format(intent=task, trajectory=trajectory_str, current_proposed_step=proposed_step)

    # Reconstruct the combined assistant text containing 
    assistant_text = f"<think>\n{rationale}\n</think>\n{target}"

    # --- Everything below here is strictly for Token Counting ---
    text_before, text_after = user_prompt.split("<image>")

    system_chat ={
            "role": "system",
            "content": system_prompt
        }
    test_prompt_chat ={
            "role": "user",
            "content": [
                {"type": "text", "text": text_before.strip()},
                {"type": "image"},
                {"type": "text", "text": text_after.strip()},
            ]
        }
    assistant_chat = {
            "role": "assistant",
            "reasoning_content": rationale,
            "content": target
        }
    test_messages = [system_chat, test_prompt_chat, assistant_chat]

    raw_prompt = processor.apply_chat_template(
        test_messages, add_generation_prompt=False, tokenize=False
    )
    image = {'bytes': current_observation}
    processed_image = process_image(image, image_patch_size=processor.image_processor.patch_size)

    # Call the processor to get the TRUE token count
    inputs = processor(
        text=[raw_prompt],
        images=[processed_image],
        return_tensors="pt"
    )
    total_tokens = len(inputs["input_ids"][0])

    if max_num_token < total_tokens:
        return None

    # -------------------------------------------------------------

    # Construct standard LLaMA-Factory ShareGPT format
    data = {
        "system": system_prompt,
        "conversations": [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": assistant_text}
        ],
        "images": [abs_image_path]
    }

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--chosen_annotation", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--rejected_annotation", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_prompt_length", type=int, default=4096)

    args = parser.parse_args()
    df_chosen = pd.read_json(args.chosen_annotation, lines=True)
    df_rejected = pd.read_json(args.rejected_annotation, lines=True)
    df_rejected = df_rejected.rename(columns={'env': 'environment', 'rejected_proposed_step': 'proposed_step'})

    df = pd.concat([df_chosen, df_rejected])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_dataset = df.copy().to_dict(orient="records")

    processor = load_processor(args.tokenizer_path)
    max_prompt_length = args.max_prompt_length

    train_data_list = [None] * len(train_dataset)

    local_save_dir = args.local_dir
    annotated_dir = os.path.join(local_save_dir, "annotated_images")
    os.makedirs(annotated_dir, exist_ok=True)

    # Wrapper function for the threads to keep track of the original row index
    def process_row(index, subset, row):
        try:
            result = process_warmup_rationale(row, subset, index, processor, max_prompt_length, annotated_dir)
            return index, result
        except Exception as e:
            print(f"Unhandled error on row {index}: {e}")
            return index, None

    # Use ThreadPoolExecutor to send multiple requests to vLLM at the same time
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        # Submit all tasks
        futures = [executor.submit(process_row, index, 'train', row) for index, row in enumerate(train_dataset)]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(train_dataset)):
            idx, result = future.result()
            train_data_list[idx] = result
    
    train_data_list = [item for item in train_data_list if item is not None]

    # Save to JSON directly instead of Parquet
    print(f"Saving {len(train_data_list)} records to train.json...")
    with open(os.path.join(local_save_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data_list, f, indent=2, ensure_ascii=False)
