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
import base64
import concurrent.futures
import glob
import io
import json
import math
import os
import random
import re
from io import BytesIO

import pandas as pd
from openai import OpenAI
from PIL import Image, ImageDraw
from tqdm import tqdm

chosen_action_rationale_system_prompt = """
You are an expert World Model and Action Critic for a GUI navigation agent.

Your task is to generate the intermediate reasoning steps and final output that a perfect Action Critic would generate naturally for a SUCCESSFUL action.

I will provide you with:
- User Instruction (Goal): The overall, high-level task the user wants the agent to accomplish.
- Trajectory: A sequential record of the agent's past proposed steps.
- Current Observation: The screenshot visual representation of the current screen. **Note: IF the agent's proposed step involves spatial coordinates, a visual marker (red 'X' or blue arrow) has been injected into this image. If the action is non-spatial (e.g., type, wait), the image is clean.**
- Current Proposed Step: The executable command (<tool_call>...</tool_call>) suggested by the agent for the current observation.
- **Target Action Mechanism**: The exact UI element interacted with.
- **Target Structural Effect**: The expected physical layout changes.
- **Target Next State**: The resulting UI state.

You must write the exact output following this strict structure:

<think>
[Act as a seamless, organic internal monologue. You must naturally deduce the 'Overall Grading: Good' by reasoning through the following aspects. You MUST separate your reasoning into three distinct paragraphs using double newlines (\\n\\n). DO NOT use explicit numbered lists, bullet points, or section headers.]

In your first paragraph, start exactly with this format: "Action String Check: [Insert raw <tool_call> string here] - Syntax is Valid." 
**CRITICAL VISUAL GROUNDING:** Next, you MUST evaluate ALL `<tool_call>` blocks in the 'Current Proposed Step' based on the arguments provided in the JSON.
- **If ANY tool call contains `coordinate` arguments (e.g., click, swipe, mouse_move):** The coordinates perfectly match the injected visual marker. Locate the **red 'X' marker** or the **blue arrow**. To prevent spatial hallucinations, you must perform a strict, literal visual scan:
    1. **Literal Center:** First, read the exact text or identify the specific icon located directly beneath the exact center intersection of the marker.
    2. **Adjacency Check:** Identify the elements immediately adjacent (left/right/above/below) to the marker.
    3. **Final Identification:** Based strictly on the literal center, explicitly describe exactly what UI element (or empty space) is targeted. (e.g., "The X is directly on the 'Subscribe' text, adjacent to the '0' counter. Therefore, it targets the Subscribe button, not the counter.")
- **If NO tool calls in the 'Current Proposed Step' contain `coordinate` arguments (e.g., ONLY scroll, type, key, wait, terminate):** You are STRICTLY FORBIDDEN from looking for or mentioning a red 'X' or blue arrow. **CRITICAL WARNING: If you see a red 'X' in the image during a non-coordinate action, it is a residual artifact from a previous step. You MUST completely ignore it.** Evaluate the global screen state and the logical parameters provided in the 'Current Proposed Step' JSON.
Secretly use the provided 'Target Action Mechanism' to ensure your visual description is perfectly accurate.
Next, analyze the 'Trajectory' against the 'User Instruction (Goal)' to establish the current progress state.
Finally, conclude this paragraph by explicitly stating what UI element the action targets in the context of this current state.

In your second paragraph, using the provided 'Target Structural Effect' and 'Target Next State', predict exactly what will physically happen to the UI layout when the specific element identified by the visual marker is interacted with. Use forward-predictive future tense (e.g., "The dropdown menu will expand..."). Focus ONLY on the structural UI skeleton changes (containers, menus, navigation) ignoring dynamic text content. CRITICAL: Do not hallucinate future steps or attribute the final completion of the goal to this specific action if it is merely a prerequisite step (e.g., do not predict text appearing if the action is merely a click to focus a field).

In your third paragraph, compare your predicted consequence directly to the 'User Instruction (Goal)'. You must explicitly link the procedural knowledge (what the agent is trying to do right now) with the state transition knowledge (the physical UI changes). Justify exactly why interacting with the element marked on the screen logically and visually completes or advances the goal, concluding that the action is 'Good'.
</think>
<criticism>
Overall Grading: Good
Explanation: [Describe the immediate physical/structural consequence of this action in the future tense. Then, explicitly state the universal UI rule or environmental mechanic that makes this interaction successful.]
</criticism>

## Action Space
{platform_action_space}

## CONSTRAINTS (STRICT):
1. NO META-LANGUAGE: Act entirely as an autonomous critic. You are STRICTLY FORBIDDEN from using phrases like "The provided label is...", "Since the target overall grading is...", or acknowledging that you were given the answer or any target state information.
2. IN-CHARACTER DEDUCTION: You must frame the diagnosis as your own organic deduction derived directly from the Current Observation (and the visual markers) and the Agent's inputs.
3. OUTPUT FORMAT: Output ONLY the <think> and <criticism> blocks.
4. NO HEADERS: Your <think> block must be a continuous stream of consciousness. You are STRICTLY FORBIDDEN from printing headers like "Intent & Execution Baseline" or "Paragraph 1". Write purely in natural paragraphs.
5. CONCEAL HIDDEN CONTEXT: You must use the 'Target' variables (Mechanism, Structural Effect, Next State) ONLY to inform your internal prediction and ensure your visual description is accurate. Inside your final <criticism> and <think> blocks, you are STRICTLY FORBIDDEN from using the word "Target" or acknowledging that these future states were provided to you. Present the predictions entirely as your own expert visual deductions.
""".strip()


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


def encode_image_to_base64(image_path):
    """Reads an image file and returns a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_parts(text):
    action_pattern = r'(?:\[|\*\*|\()Action Mechanism(?:\]|\*\*|\))\s*:\s*(.*?)(?=\n(?:\[|\*\*|\()|$)'
    structural_pattern = r'(?:\[|\*\*|\()Structural Effect(?:\]|\*\*|\))\s*:\s*(.*?)(?=\n(?:\[|\*\*|\()|$)'
    
    action = re.search(action_pattern, text, re.DOTALL)
    structural = re.search(structural_pattern, text, re.DOTALL)
    
    return (
        action.group(1).strip() if action else "",
        structural.group(1).strip() if structural else ""
    )

def mark_action_on_base64_image(action_str: str, base64_img: str, debug_save_path: str = None) -> str:
        # 1. Use findall to get every <tool_call> block in the string
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(action_str)

        if not matches:
            return base64_img

        # Differentiate between single-point clicks and two-point drags/swipes
        point_actions = {
            "left_click", "click", "right_click", "double_click", 
            "triple_click", "middle_click", "mouse_move", "long_press"
        }
        drag_actions = {
            "swipe", "left_click_drag"
        }

        # 2. Parse all blocks and collect valid shapes to draw
        shapes_to_draw = []
        for match_text in matches:
            try:
                tool_call = ast.literal_eval(match_text.strip())
                args = tool_call.get("arguments", {})
                action_type = args.get("action")
                
                # Handle single-point actions (draws an 'X')
                if action_type in point_actions and "coordinate" in args and isinstance(args["coordinate"], list):
                    shapes_to_draw.append(("point", args["coordinate"]))
                
                # Handle two-point actions (draws an Arrow)
                elif action_type in drag_actions and "coordinate" in args and "coordinate2" in args:
                    if isinstance(args["coordinate"], list) and isinstance(args["coordinate2"], list):
                        shapes_to_draw.append(("arrow", args["coordinate"], args["coordinate2"]))

            except (ValueError, SyntaxError):
                continue

        if not shapes_to_draw:
            return base64_img

        # 3. Decode the base64 string into a PIL Image ONCE
        try:
            img_data = base64.b64decode(base64_img)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        except Exception as e:
            print(f"Image decoding failed: {e}")
            return base64_img

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

        # 4. Loop through and draw the shapes
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

        # --- DEBUG SAVE FEATURE ---
        if debug_save_path:
            try:
                img.save(debug_save_path)
                print(f"Debug image saved to: {debug_save_path}")
            except Exception as e:
                print(f"Failed to save debug image: {e}")

        # 5. Encode the modified PIL image back to a base64 string
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        annotated_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return annotated_base64

def extract_tool_call_only(text):
    if not isinstance(text, str):
        return text
        
    # re.DOTALL ensures the '.' matches newlines as well
    match = re.search(r'(<tool_call>.*?</tool_call>)', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text # Fallback in case a row is formatted strangely


def add_mouse_move_before_scroll(text, prob=0.5):
    mouse_move_block = '''<tool_call>\n{"name": "computer_use", "arguments": {"action": "mouse_move", "coordinate": [500, 554]}}\n</tool_call>\n'''

    # Pattern captures the entire tool_call block containing "scroll"
    pattern = r'(<tool_call>.*?"action"\s*:\s*"scroll".*?</tool_call>)'

    def replacer(match):
        block = match.group(1)
        if random.random() < prob:
            return mouse_move_block + block
        return block

    return re.sub(pattern, replacer, text, flags=re.DOTALL)



def warmup_rationale(task_sample):
    task = task_sample.get("task", "")
    trajectory = task_sample.get("trajectory", [])
    proposed_step = task_sample.get("proposed_step", "")

    image_path = task_sample.get("image_path", "")
    base64_image = encode_image_to_base64(image_path)

    transition_rationale = task_sample.get("transition_rationale", "")
    action_mechanism, structural_effect = extract_parts(transition_rationale)
    state_transition = task_sample.get("state_transition", "")

    if task_sample['environment'] == 'android':
        platform = 'mobile'
    elif task_sample['environment'] in ['web', 'ubuntu', 'windows', 'mac']:
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

    # mark
    base64_image = mark_action_on_base64_image(proposed_step, base64_image)

    system_prompt = chosen_action_rationale_system_prompt.format(platform_action_space=action_spaces[platform])

    user_prompt = f"User Instruction (Goal):\n{task}\n\n"
    user_prompt += f"Trajectory:\n{trajectory_str}\n\n"
    user_prompt += f"Current Observation:\n<image>\n\n"

    user_prompt += f"Current Proposed Step: {proposed_step}\n\n"

    user_prompt += f"**Target Action Mechanism**:\n{action_mechanism}\n\n"
    user_prompt += f"**Target Structural Effect**:\n{structural_effect}\n\n"
    user_prompt += f"**Target Next State**:\n{state_transition}\n\n"

    text_before, text_after = user_prompt.split("<image>")

    model_name = "Qwen/Qwen3-VL-32B-Thinking"
    attempt = 0

    rationale = None
    while attempt < 5:
        attempt += 1
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": text_before.strip()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": text_after.strip()},
                ]}
            ]

            response = client_vllm.chat.completions.create(# type: ignore
                model=model_name,
                messages=messages,
            )

            rationale = f"<think>\n{response.choices[0].message.reasoning.strip()}\n</think>\n{response.choices[0].message.content.strip()}"
            if rationale is not None:
                break

        except Exception as e:
            print(f"[Attempt {attempt}] Error generating rationale: {e}")

    return rationale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", default=None)
    parser.add_argument("--local_dir", default=None)

    args = parser.parse_args()

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    output_path = os.path.join(local_save_dir, "critic_chosen_dataset_fixed_grounding_no_verbal.jsonl")

    annotation_dir = args.annotation_dir
    df = pd.read_json(os.path.join(annotation_dir, "double_blind_filtered_chosen_samples.jsonl"), lines=True)

    df = df.drop_duplicates(subset=["image_path"])

    df['state_transition'] = df['state_transition'].fillna("N/A")
    df['transition_rationale'] = df['transition_rationale'].fillna("N/A")

    # Add more variation on the vector error
    df_scroll = df[df["proposed_step"].astype(str).str.contains(
            r'"action"\s*:\s*"scroll"', 
            case=False, 
            na=False,
            regex=True
        )
    ]
    df_other = df[~df["proposed_step"].astype(str).str.contains(
            r'"action"\s*:\s*"scroll"',
            case=False,
            na=False,
            regex=True
        )
    ]
    df_scroll["proposed_step"] = df_scroll["proposed_step"].apply(
        lambda x: add_mouse_move_before_scroll(str(x), prob=0.3)
    )
    df = pd.concat([df_other, df_scroll]).sample(frac=1).reset_index(drop=True)

    # Calculate exactly how many rows make up 30% of the total dataset
    num_no_intent_samples = int(len(df) * 0.30)

    # Sample the 30% strictly from the safe subset
    df_no_intent = df.sample(n=num_no_intent_samples, random_state=42)
    df = df_no_intent
    df['proposed_step'] = df['proposed_step'].apply(extract_tool_call_only)

    # ==========================================

    client_vllm = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    warmup_data_list = df.to_dict(orient="records")
    annotated_results = [None] * len(warmup_data_list)

    def process_row(index, row):
        try:
            result = warmup_rationale(row)
            return index, result
        except Exception as e:
            print(f"Unhandled error on row {index}: {e}")
            return index, None

    # Execute concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_row, idx, sample) 
            for idx, sample in enumerate(warmup_data_list)
        ]

        # Use tqdm to track completed threads
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(warmup_data_list)):
            idx, result = future.result()
            annotated_results[idx] = result

    df['target_output'] = annotated_results
    df_final = df[df['target_output'].notna()]

    print(f"Successfully generated {len(df_final)} annotated samples out of {len(df)}.")
    df_final.to_json(output_path, orient="records", lines=True)
    # df_grpo.to_json(remaining_input_path, orient="records", lines=True)
    print("done")

