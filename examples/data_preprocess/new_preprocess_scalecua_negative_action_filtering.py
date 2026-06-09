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
import json
import os
import random
import re

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

double_blind_prompt = """
You are an expert Evaluator for GUI Agent training data.

### Context Provided:
- User Instruction (Goal): The overall, high-level task the user wants the agent to accomplish.
- Trajectory: A sequential record of the agent's past proposed steps.
- Current Observation: The screenshot visual representation of the current screen.
- Candidate Proposed Step A: a possible next step the agent could take.
- Candidate Proposed Step B: another possible next step the agent could take.

### Evaluation Task:
1. **Preference Check:** One of them is the correct, optimal step. The other is a flawed step representing a specific type of error. Which candidate (A or B) is the definitively better, correct action to progress the task?
2. **Error Dimension Prediction:** Look at the *worse* candidate. What specific Error Dimension from the list above does it represent? If the worse candidate is just a valid alternative, or if it doesn't clearly fit any of the strictly defined dimensions, you must classify it as "Invalid".

### Action Space
{platform_action_space}

### Available Error Dimensions:
{dimensions_list_str}

### Output Format:
First, use a <think> block to step-by-step reason through the comparison. You MUST structure your thinking in this exact order:
1. **Syntax Check:** You must act as a strict code parser. For BOTH candidates, you MUST extract and copy the exact JSON `<tool_call>` string character-by-character inside backticks before evaluating it. Use this exact format:
"Action A Raw: `[Insert exact <tool_call> string]` - Syntax: [Valid/Invalid] - [Reason]"
"Action B Raw: `[Insert exact <tool_call> string]` - Syntax: [Valid/Invalid] - [Reason]"
Explicitly check if required keyword arguments are missing, if wrapper tags are omitted, or if brackets/quotes are malformed based on the Action Space.
2. **Semantic & Grounding Check:** Compare their intended targets and coordinates against the visual elements in the Current Observation.
3. **Diagnosis:** Determine which is the correct step and which is flawed. Explicitly map the flawed step to one of the provided Error Dimensions.

After the <think> block, output ONLY a strict JSON object:
```json
{{
    "better_candidate": "A", // Or "B"
    "predicted_error_dimension": "[Insert the EXACT top-level name of the selected Error Dimension (e.g., 'Termination Misjudgment')]",
    "reasoning_summary": "[Brief 1-sentence summary of your evaluation]"
}}
```
""".strip()


error_dimensions = [
    "**Grounding/Spatial Error:** The agent's semantic intent is correct, but coordinate precision fails. It physically near-misses the target, landing in adjacent dead or incorrect space.",
    "**Procedural Prerequisite Neglect:** The agent attempts an operation but skips a mandatory preceding interface state change. This includes (A) Logical Neglect (failing to execute a required preparatory action to make an element receptive to input), or (B) Spatial/Occlusion Neglect (failing to dismiss a foreground overlay physically blocking the target on the Z-axis).",
    "**Semantic Error:** The agent targets perfectly, but fundamentally misinterprets the vocabulary, icons, or UI paradigms, clicking the wrong element for the goal (e.g., clicking a deceptive ad or the wrong sign-in button).",
    "**Termination Misjudgment:** The agent misjudges task completion based on the visual state. It prematurely outputs 'terminate', incorrectly outputs a 'terminate' action BEFORE using the 'answer' tool when the overall goal requires explicitly reporting the text back to the user (Reporting Failure), or hallucinates redundant steps after the overall goal is already met instead of outputing 'answer' or 'terminate'.",
    "**Constraint Neglect:** The user instruction explicitly dictates multifaceted requirements. The agent executes an action that completely ignores these explicit text constraints by selecting a visually available, highly distracting alternative. For example, if the instruction says 'click the 2012 report', the agent clicks the '2023' report visible on the screen.",
    "**Action Formulation Error:** The agent's intent is correct, but the `<tool_call>` JSON string crashes the parser due to syntax errors, missing wrapper tags, malformed code, or invalid/omitted arguments.",
    "**Observation Neglect:** The agent attempts to search, scroll, or open menus to find information that is already clearly visible on the current screen.",
    "**Suboptimal Path:** Progresses the task but selects a highly inefficient sequence of micro-actions (e.g., repetitive arrow clicks or backspaces) when a faster standard UI paradigm is available.",
    "**Parameter Vector Miscalibration:** The agent chooses the correct interaction type (e.g., scroll, swipe, drag) but critically fails the physical vector execution. Look for a contradiction: the agent reasons or executes the exact opposite direction of the goal (Polarity Reversal) or uses a drastically undersized scale resulting in negligible UI movement (Magnitude Insufficiency).",
    "**Visual Hallucination:** The agent interacts with a UI element that strictly does not exist on the current screen. It either clicks a 'ghost' element from a previous state that has since disappeared, or blindly guesses a coordinate based on standard UI layouts without visual confirmation.",
    "**Timing and Latency Neglect:** The agent executes a logically correct action prematurely, ignoring visual indicators (e.g., loading spinners, progress bars, unfolding menus) that the system is busy or mid-transition.",
    "**Action-Operation Misalignment:** The agent's verbalized chain-of-thought and planned intent are logically sound, but its executable `<tool_call>` drastically contradicts this intent. It writes a highly logical 'Action:' description, but the actual JSON tool call it generates is completely different and nonsensical for that stated goal."
]


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


def map_old_action_to_agent_action(action_str: str, operation: str, platform: str) -> str:
    """
    Parses the old python-style action string from the dataframe and maps it
    to the new JSON <tool_call> format based on the platform.
    Supports multiple actions separated by newlines (e.g. click()\nwrite()).
    """
    if pd.isna(action_str) or not action_str.strip():
        return ""

    # 1. Parse Python string to AST using "exec" to support multiple lines
    try:
        tree = ast.parse(action_str.strip(), mode="exec")
    except Exception:
        # Fallback if entirely unparseable (to prevent crashing the pipeline)
        return f"<tool_call>\n{{\"name\": \"{'mobile_use' if platform == 'mobile' else 'computer_use'}\", \"arguments\": {{\"action\": \"wait\", \"time\": 1}}}}\n</tool_call>"

    tool_name = "mobile_use" if platform == "mobile" else "computer_use"
    mapped_tool_calls = []

    # 2. Iterate through every line/statement in the parsed code
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue

        call = node.value
        func_name = call.func.id if isinstance(call.func, ast.Name) else ""
        
        # Reset args and kwargs for EACH function call
        args = []
        kwargs = {}

        for arg in call.args:
            try:
                args.append(ast.literal_eval(arg))
            except Exception:
                pass
        for kw in call.keywords:
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                pass

        arguments = {}

        # Safely convert coordinates to ints if they exist
        if "x" in kwargs and "y" in kwargs:
            kwargs["x"] = int(kwargs.get("x") or 0)
            kwargs["y"] = int(kwargs.get("y") or 0)

        # --- MAP COMPUTER/WEB ---
        if platform in ["desktop", "web", "windows", "mac", "ubuntu"]:
            if func_name in ["click", "left_click"]:
                arguments = {"action": "left_click"}
                if "x" in kwargs and "y" in kwargs:
                    arguments["coordinate"] = [kwargs["x"], kwargs["y"]]
                
                # Catch modifiers
                if kwargs.get("clicks", 1) == 2:
                    arguments["action"] = "double_click"
                elif kwargs.get("clicks", 1) >= 3:
                    arguments["action"] = "triple_click"
                elif kwargs.get("button") == "right":
                    arguments["action"] = "right_click"
                elif kwargs.get("button") == "middle":
                    arguments["action"] = "middle_click"
            
            elif func_name == "tripleClick":
                arguments = {"action": "triple_click"}
                if "x" in kwargs and "y" in kwargs:
                    arguments["coordinate"] = [kwargs["x"], kwargs["y"]]
            
            elif func_name == "doubleClick":
                arguments = {"action": "double_click"}
                if "x" in kwargs and "y" in kwargs:
                    arguments["coordinate"] = [kwargs["x"], kwargs["y"]]

            elif func_name == "rightClick":
                arguments = {"action": "right_click"}
                if "x" in kwargs and "y" in kwargs:
                    arguments["coordinate"] = [kwargs["x"], kwargs["y"]]
                    
            elif func_name == "moveTo":
                arguments = {"action": "mouse_move"}
                if "x" in kwargs and "y" in kwargs:
                    arguments["coordinate"] = [kwargs["x"], kwargs["y"]]
                    
            elif func_name == "dragTo":
                arguments = {"action": "left_click_drag"}
                if "x" in kwargs and "y" in kwargs:
                    arguments["coordinate"] = [kwargs["x"], kwargs["y"]]
                    
            elif func_name == "scroll":
                # Extract clicks (check kwargs first, then positional args, default to 10)
                clicks = kwargs.get("clicks", args[0] if args else 10)
                
                # Convert physical wheel clicks to screen pixels (1 click ≈ 100 pixels)
                pixels = int(clicks * 100)
                
                arguments = {"action": "scroll", "pixels": pixels}

            elif func_name == "swipe":
                direction = kwargs.get("direction", "up")
                amount = kwargs.get("amount", 0.5)
                pixels = int(amount * 1000) if direction in ["down", "right"] else int(amount * -1000)

                if direction in ["left", "right"]:
                    arguments = {"action": "hscroll", "pixels": pixels}
                else:
                    arguments = {"action": "scroll", "pixels": pixels}

            elif func_name in ["write", "type"]:
                arguments = {"action": "type", "text": kwargs.get("message", args[0] if args else "")}
                
            elif func_name in ["press", "hotkey", "keyDown", "keyUp"]:
                keys_raw = kwargs.get("keys", kwargs.get("key", args))
                
                # Normalize whatever we extracted into a flat list of strings
                if isinstance(keys_raw, str):
                    keys_list = [keys_raw]
                elif isinstance(keys_raw, (list, tuple)):
                    keys_list = list(keys_raw)
                else:
                    keys_list = [] # Fallback for unexpected data types
                arguments = {"action": "key", "keys": list(keys_list)}
                
            elif func_name == "wait":
                arguments = {"action": "wait", "time": kwargs.get("seconds", 3)}
                
            elif func_name in ["response", "answer"]:
                arguments = {"action": "answer", "text": kwargs.get("answer", args[0] if args else "")}
                
            elif func_name == "terminate":
                arguments = {"action": "terminate", "status": kwargs.get("status", "success")}

        # --- MAP MOBILE ---
        elif platform == "mobile":
            if func_name == "click":
                arguments = {"action": "click"}
                if "x" in kwargs and "y" in kwargs:
                    arguments["coordinate"] = [kwargs["x"], kwargs["y"]]
                    
            elif func_name == "long_press":
                arguments = {"action": "long_press"}
                if "x" in kwargs and "y" in kwargs:
                    arguments["coordinate"] = [kwargs["x"], kwargs["y"]]
                arguments["time"] = kwargs.get("duration", 1)
                
            elif func_name == "swipe":
                arguments = {"action": "swipe"}
                if "from_coord" in kwargs and "to_coord" in kwargs:
                    arguments["coordinate"] = [int(kwargs["from_coord"][0]), int(kwargs["from_coord"][1])]
                    arguments["coordinate2"] = [int(kwargs["to_coord"][0]), int(kwargs["to_coord"][1])]
                    
            elif func_name == "open_app":
                arguments = {"action": "open", "text": kwargs.get("app_name", args[0] if args else "")}
                
            elif func_name == "navigate_home":
                arguments = {"action": "system_button", "button": "Home"}
                
            elif func_name == "navigate_back":
                arguments = {"action": "system_button", "button": "Back"}
                
            elif func_name in ["write", "type"]:
                arguments = {"action": "type", "text": kwargs.get("message", args[0] if args else "")}
                
            elif func_name in ["press", "key", "hotkey"]:
                arguments = {"action": "key", "text": kwargs.get("keys", args[0] if args else "")}
                
            elif func_name == "wait":
                arguments = {"action": "wait", "time": kwargs.get("seconds", 3)}
                
            elif func_name in ["response", "answer"]:
                arguments = {"action": "answer", "text": kwargs.get("answer", args[0] if args else "")}
                
            elif func_name == "terminate":
                arguments = {"action": "terminate", "status": kwargs.get("status", "success")}

        # Fallback if action wasn't mapped
        if not arguments:
            arguments = {"action": "wait", "time": 1}

        # Format and append this specific action
        json_payload = {
            "name": tool_name,
            "arguments": arguments
        }
        mapped_tool_calls.append(f"<tool_call>\n{json.dumps(json_payload)}\n</tool_call>")

    # 3. Return all mapped calls joined by newlines (or fallback if empty)
    if not mapped_tool_calls:
         return f"<tool_call>\n{{\"name\": \"{tool_name}\", \"arguments\": {{\"action\": \"wait\", \"time\": 1}}}}\n</tool_call>"

    return "\n".join(mapped_tool_calls)


def encode_image_to_base64(image_path):
    """Reads an image file and returns a base64 encoded string."""
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found.")
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_json(text):
    start = text.find("{")
    if start == -1:
        return None

    brace_count = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        if char == '"' and not escape:
            in_string = not in_string

        if char == "\\" and not escape:
            escape = True
            continue
        else:
            escape = False

        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

            if brace_count == 0:
                json_str = text[start:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None

    return None


def validate_negative_sample_double_blind(row, client):
    """
    Evaluates the row by randomly assigning the chosen and rejected samples to Option A and B,
    forcing the model to blindly predict which is better and what error the worse one represents.
    """
    image_path = row['image_path']
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    if row['env'] == 'android':
        platform = 'mobile'
    elif row['env'] in ['web', 'ubuntu', 'windows', 'mac']:
        platform = 'desktop'
    else:
        ValueError(f"{row['env']} is not defined.")

    random.shuffle(error_dimensions)
    dimensions_list_str = "\n".join(error_dimensions)

    system_prompt = double_blind_prompt.format(dimensions_list_str=dimensions_list_str,platform_action_space=action_spaces[platform])

    trajectory_str = ""
    for i, step in enumerate(row['trajectory']):
        trajectory_str += (
            f"Step {i+1}:\n"
            f"Action: {step.get('operation')}\n"
            f"{step.get('action')}\n\n"
        )
    trajectory_str = trajectory_str.strip()

    # Format the two samples
    chosen_text = (
        f"Proposed Step: Action: {row['operation']}\n{row['action']}\n"
    )
    
    rejected_text = (
        f"Proposed Step: {row['rejected_proposed_step']}\n"
    )

    # Randomly assign the ground-truth and negative sample to A or B
    is_a_chosen = random.choice([True, False])
    if is_a_chosen:
        option_a_text = chosen_text
        option_b_text = rejected_text
        expected_better = "A"
    else:
        option_a_text = rejected_text
        option_b_text = chosen_text
        expected_better = "B"

    user_content_text = (
        f"User Instruction (Goal):\n{row['task']}\n\n"
        f"Trajectory:\n{trajectory_str}\n\n"
        f"Current Observation:\n"
    )
    user_option_text = f"Candidate Proposed Step A:\n{option_a_text}\n\nCandidate Proposed Step B:\n{option_b_text}"

    attempt = 0
    final_json = None
    model_name = "Qwen/Qwen3-VL-32B-Thinking"

    while attempt < 3:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_content_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": user_option_text},
                    ]}
                ],
            )
            
            parsed_json = extract_json(response.choices[0].message.content.strip())
            
            # Validate output keys
            if parsed_json and "better_candidate" in parsed_json and "predicted_error_dimension" in parsed_json:
                final_json = parsed_json
                final_json['expected_better'] = expected_better # Save the ground truth label for later verification
                break
                
        except Exception as e:
            print(f"[Attempt {attempt}] Error validating row {row.get('step_index')}: {e}")

    return final_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./output")
    args = parser.parse_args()

    local_save_dir = args.local_dir
    input_path = os.path.join(local_save_dir, 'negative_samples.jsonl')

    output_path = os.path.join(local_save_dir, 'double_blind_filtered_negative_samples.jsonl')
    output_chosen_path = os.path.join(local_save_dir, 'double_blind_filtered_chosen_samples.jsonl')

    client_vllm = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )

    df = pd.read_json(input_path, lines=True)
    validation_results = [None] * len(df)
    
    def process_validation(index, row):
        try:
            result = validate_negative_sample_double_blind(row, client_vllm)
            return index, result
        except Exception as e:
            print(f"Unhandled error on row {index}: {e}")
            return index, None

    print("Starting double-blind prediction validation phase...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_validation, index, row) for index, row in df.iterrows()]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df)):
            idx, result = future.result()
            validation_results[idx] = result

    # Unpack structured results back into the DataFrame
    predicted_dims = []
    better_candidates = []
    expected_betters = []
    conf_scores = []
    reasons = []

    for res in validation_results:
        if res is not None:
            # Clean up the output to strictly "A" or "B"
            better_opt = str(res.get("better_candidate", "")).strip().upper()
            if "A" in better_opt: 
                better_opt = "A"
            elif "B" in better_opt:
                better_opt = "B"
            
            better_candidates.append(better_opt)
            expected_betters.append(res.get("expected_better", ""))
            predicted_dims.append(res.get("predicted_error_dimension", "API Failure"))
            reasons.append(res.get("reasoning_summary", ""))
        else:
            better_candidates.append("FAIL")
            expected_betters.append("NONE")
            predicted_dims.append("API Failure")
            reasons.append("API Failure")

    df['judge_better_candidate'] = better_candidates
    df['expected_better_candidate'] = expected_betters
    df['predicted_dimension'] = predicted_dims
    df['evaluator_reasoning'] = reasons

    # Extract the clean string of the original target dimension from the synthesizer
    # Handles cases where the synthesizer outputs "**Semantic Error**"
    df['clean_target_dimension'] = df['dimension_violated'].apply(
            lambda x: str(x).split(":")[0].replace("*", "").strip()
        )
    df['clean_predicted_dimension'] = df['predicted_dimension'].apply(
            lambda x: str(x).split(":")[0].replace("*", "").strip()
        )

    original_len = len(df)
    
    # --- MASKS ---
    # 1. Did the Ground Truth win? (Is the chosen action fundamentally sound?)
    ground_truth_won_mask = df['judge_better_candidate'] == df['expected_better_candidate']
    
    # 2. Did the Error Dimension match? (Is the negative sample high-quality and unambiguous?)
    dimension_matched_mask = df['clean_predicted_dimension'] == df['clean_target_dimension']

    # --- BUCKET 1: Pristine Pairs ---
    df_pristine_pairs = df[ground_truth_won_mask & dimension_matched_mask].copy()

    df_valid_chosen = df[ground_truth_won_mask].copy()
    df_valid_chosen = df_valid_chosen.drop(columns=['dimension_violated', 'rejected_thought', 'rejected_proposed_step', 'clean_target_dimension'], errors='ignore')

    # --- LOGGING ---
    print(f"\n--- Double-Blind Validation Filtering Summary ---")
    print(f"Original samples evaluated: {original_len}")
    print(f"1. Pristine Pairs (Perfect Pos/Neg match): {len(df_pristine_pairs)}")
    print(f"2. Valid Chosen Actions: {len(df_valid_chosen)}")

    # Clean up the temporary tracking columns before saving
    eval_cols = ['judge_better_candidate', 'expected_better_candidate', 'clean_predicted_dimension']
    df_pristine_pairs = df_pristine_pairs.drop(columns=eval_cols + ['clean_target_dimension'], errors='ignore')
    df_valid_chosen = df_valid_chosen.drop(columns=eval_cols, errors='ignore')

    df_pristine_pairs.to_json(
            output_path,
            orient='records', lines=True, force_ascii=False
        )

    df_valid_chosen.to_json(
            output_chosen_path,
            orient='records', lines=True, force_ascii=False
        )

    print(f"Strictly curated dataset saved to {output_path}")