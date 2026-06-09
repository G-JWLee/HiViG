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

SUMMARY_SYSTEM_PROMPT = """
You are the Macro-Strategy and Reflection module for a GUI navigation agent. Your task is to analyze the agent's recent action, perform a visual diff, and synthesize a strategic summary that explicitly connects the current visual state to the User Instruction.

### Task Description
You will be provided with:
- User Instruction (Goal): The overall, high-level task the user wants the agent to accomplish.
- Previous Summary: The memory bank from the last turn.
- Previous Observation: The screenshot visual representation of the screen BEFORE the proposed step.
- Previous Proposed Step: The combined verbalized intent (Action: ...) and the executable command (<tool_call>...</tool_call>) previously suggested by the agent for the previous observation.
- Current Observation: The screenshot visual representation of the screen AFTER the proposed step.

Your task is to act as a summary model. You must determine if the agent's current trajectory is actively converging on the User Instruction.

### Action Space
{platform_action_space}

### Output Format
You must strictly follow this exact structure. Before outputting the final summary, you must reason through the visual state and logical deduction step-by-step inside a <think> block.

<think>
In your monologue, you must naturally flow through these logical steps:

- Perform a fast visual diff. State the 'Previous Proposed Step', compare the 'Previous Observation' to the 'Current Observation', and explicitly describe exactly what changed physically on the screen. Definitively conclude whether the 'Previous Proposed Step' resulted in a UI update or a silent failure.

- Map the verified 'Current Observation' to the 'User Instruction (Goal)'. Identify the visual gap: what goal-related UI elements are currently present, and what are missing?
</think>
<summary>
Progress Summary: [LONG TERM MEMORY: Synthesize the 'Previous Summary' with the current result. Compress completed micro-steps into dense macro-achievements (e.g., "Successfully logged in"). Only append the new action if it succeeded.]
Last Action Result: [SHORT TERM MEMORY: An objective, static description of exactly what the last action accomplished visually.]
</summary>
""".strip()


SUMMARY_USER_PROMPT = """
User Instruction (Goal):
{intent}

Previous Summary:
{previous_summary}

Previous Observation:
<image>

Previous Proposed Step:
{proposed_step}

Current Observation:
<image>
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


def clean_summary(summary):
    if not isinstance(summary, str):
        return ""

    summary = summary.strip()

    if summary.startswith("Progress Summary:"):
        summary = summary[len("Progress Summary:"):].strip()

    return summary


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
    try:
        task = task_sample.get("task")

        # Note that we already saved the marked image
        prev_img_path = task_sample.get("prev_image_path")
        curr_img_path = task_sample.get("current_image_path")

        proposed_step = task_sample.get("proposed_step")

        previous_observation = load_image_bytes(prev_img_path)
        current_observation = load_image_bytes(curr_img_path)

        abs_prev_img_path = os.path.abspath(prev_img_path)
        abs_curr_img_path = os.path.abspath(curr_img_path)

        rationale = task_sample.get("summary_rationale")
        target_summary = task_sample.get("generated_summary")
        previous_summary = task_sample.get("previous_summary")

        if task_sample['environment'] == 'android':
            platform = 'mobile'
        elif task_sample['environment'] in ['web', 'windows', 'mac']:
            platform = 'desktop'
        else:
            ValueError(f"{task_sample['environment']} is not defined.")
        
        if rationale is None or target_summary is None:
            return None

        system_prompt = SUMMARY_SYSTEM_PROMPT.format(platform_action_space=action_spaces[platform])
        user_prompt = SUMMARY_USER_PROMPT.format(intent=task, previous_summary=previous_summary, proposed_step=proposed_step)

        # Reconstruct the combined assistant text containing 
        assistant_text = f"<think>\n{rationale}\n</think>\n<summary>\n{target_summary}\n</summary>"

        # --- Everything below here is strictly for Token Counting ---
        text_before, text_middle, text_after = user_prompt.split("<image>")

        system_chat ={
                "role": "system",
                "content": system_prompt
            }
        test_prompt_chat ={
                "role": "user",
                "content": [
                    {"type": "text", "text": text_before},
                    {"type": "image"},
                    {"type": "text", "text": text_middle},
                    {"type": "image"},
                    {"type": "text", "text": text_after},
                ]
            }
        assistant_chat = {
                "role": "assistant",
                "reasoning_content": rationale,
                "content": target_summary
            }
        test_messages = [system_chat, test_prompt_chat, assistant_chat]

        raw_prompt = processor.apply_chat_template(
            test_messages, add_generation_prompt=False, tokenize=False
        )

        previous_image = {'bytes': previous_observation}
        processed_previous_image = process_image(previous_image, image_patch_size=processor.image_processor.patch_size)

        current_image = {'bytes': current_observation}
        processed_current_image = process_image(current_image, image_patch_size=processor.image_processor.patch_size)

        # Call the processor to get the TRUE token count
        inputs = processor(
            text=[raw_prompt],
            images=[processed_previous_image, processed_current_image],
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
            "images": [abs_prev_img_path, abs_curr_img_path]
        }

        return data
    except Exception as e:
        print(f"Unhandled error on row {idx}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--summary_annotation", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_prompt_length", type=int, default=4096)

    args = parser.parse_args()
    df = pd.read_json(args.summary_annotation, lines=True)

    train_dataset = df.copy().to_dict(orient="records")

    processor = load_processor(args.tokenizer_path)
    max_prompt_length = args.max_prompt_length

    train_data_list = [None] * len(train_dataset)

    local_save_dir = args.local_dir
    annotated_dir = os.path.join(local_save_dir, "annotated_summary_images")
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
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
