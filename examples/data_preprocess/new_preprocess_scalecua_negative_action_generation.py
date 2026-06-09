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
from collections import defaultdict

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

negative_sample_generation_prompt = """
You are an expert GUI Agent Data Synthesizer.

### Context Provided:
- User Instruction (Goal): The overall, high-level task the user wants the agent to accomplish.
- Trajectory: A sequential record of the agent's past proposed steps.
- Current Observation: The screenshot visual representation of the current screen.
- Chosen Thought: The agent's step-by-step reasoning that justifies the subsequent proposed step.
- Chosen Proposed Step: The combined verbalized intent (Action: ...) and the executable command (<tool_call>...</tool_call>) suggested by the agent for the current observation.

Your task is to generate high-quality "rejected samples" (plausible but incorrect thoughts and proposed steps) to train an advanced GUI Critic Model. Given a user's instruction, the trajectory, the ground-truth `Chosen Thought` and `Chosen Proposed Step`, and the current visual observation of the screen, you must synthesize a plausible `rejected_thought` and `rejected_proposed_step`. These rejected samples must mimic the specific, complex failure modes of frontier GUI agents.

### Action Space
{platform_action_space}

### Core Generation Constraints
1. **Absolute Divergence (CRITICAL):** You MUST accept the provided `Chosen Proposed Step` as the flawless, indisputable correct action, even if it contradicts your visual analysis. The executable `<tool_call>` within your `rejected_proposed_step` MUST BE STRICTLY DIFFERENT from the ground truth. If you output the exact same tool call string, you have failed.
2. **No Shortcuts:** The rejected step CANNOT be a "different but correct" way to solve the task (e.g., typing a direct URL instead of using a menu). It must be a genuine mistake.
3. **Temporal Alignment:** You are generating an ALTERNATIVE to the `Chosen Proposed Step` for the *current* step only. You are synthesizing what the flawed agent *would do instead* right now.
4. **Logic Isolation:** The `rejected_thought` must read as if the agent is completely confident in its mistake. Inside the `rejected_thought` string, never mention, compare, or contrast it with the correct path. However, *as the synthesizer*, you must actively use the `Chosen Proposed Step` as an anti-target to ensure your generated action is physically different and fully committed to the selected error dimension.
5. **The Termination Paradox:** If the `Chosen Proposed Step` is `terminate`, you face a logic trap. You CANNOT output `terminate` as your rejected action. You must instead simulate **Over-Execution of Termination Misjudgment** by generating a physical UI interaction (e.g., a redundant click, scroll, or type) because the flawed agent failed to realize the task was already complete.

### Generation Guidelines (Target Error Dimensions)
Analyze the State & Trajectory deeply. Synthesize a rejected sample that perfectly mimics ONE of the targeted error dimensions below:
{three_dimensions}

### Output Format:
First, use a <think> block to reason through the negative sample generation step-by-step. You MUST structure your thinking process as follows:
1. **Visual & Trajectory Anchoring:** Briefly summarize the visible UI elements in the Current Observation. Identify what just happened in the Trajectory and what the `Chosen Proposed Step` is attempting to do.
2. **Dimension Selection & Alignment:** Explicitly choose ONE Error Dimension that most naturally fits the current UI state to create a strong hard negative. Explain exactly how your planned action will manifest this specific error.
3. **Drafting the Flaw:** Draft the 'rejected_thought'. It must read like the confident internal reasoning of a capable GUI agent. Ground the reasoning in the visible UI elements, but introduce a subtle misinterpretation or incorrect assumption that aligns with the selected error dimension. The agent's incorrect assumption MUST be rooted in standard, realistic software design patterns. Do NOT express uncertainty, hesitation, or mention that this is a rejected sample.
4. **Explicit Divergence Proof (CRITICAL):** Before drafting the final JSON, you MUST explicitly write down the ground-truth `<tool_call>`. Then, write down your planned rejected `<tool_call>`. You MUST explicitly state the mathematical or structural difference between them (e.g., "The chosen action clicks [100, 200], so I will subtract 15 pixels and click [85, 200]", or "The chosen action is valid JSON, so I will intentionally delete the closing bracket to crash the parser").

After the <think> block, output a strict JSON object confirming the variables:
```json
{{
    "dimension_violated": "[Insert the EXACT top-level name of the selected Error Dimension (e.g., 'Procedural Prerequisite Neglect')]",
    "rejected_thought": "[The internal reasoning the agent *would have generated* to conclude with the bad action. MUST begin with a brief analysis of visible UI elements from the Current Observation before forming the decision and match the format of the `Chosen Thought`]",
    "rejected_proposed_step": "Action: [The verbalized description of the bad action]\\n<tool_call>\\n{{\\"name\\": \\"computer_use\\", \\"arguments\\": {{\\"action\\": \\"left_click\\", \\"coordinate\\": [500, 500]}}}}\\n</tool_call>"
}}
```
""".strip()


error_dimensions = [
    "**Procedural Prerequisite Neglect:** Attempts a logical operation but skips a mandatory preceding interface state change. This includes logical neglect, such as skipping a required interaction rule (e.g., compressing files without selecting them first, typing without clicking to focus), as well as spatial or occlusion neglect, where the agent attempts to interact with a background element without first dismissing a foreground overlay (like a modal, pop-up, or sticky menu) blocking the target on the Z-axis.",
    "**Semantic Error:** Perfectly targets a valid, clickable element, but fundamentally misunderstands the inherent meaning, function, or consequence of that specific UI element in relation to the goal. The error stems purely from misinterpreting vocabulary, icons, or standard UI paradigms (e.g., clicking 'Sign Up' when trying to log into an existing account, or clicking a deceptive ad disguised as a download button).",
    "**Termination Misjudgment:** The agent fundamentally misunderstands the completion status of the User Instruction (Goal) based on the visual state. A critical manifestation of this is a reporting failure, where the agent successfully locates the required information but incorrectly outputs a 'terminate' action instead of using the 'answer' tool when the overall goal requires explicitly reporting the text back to the user. This dimension also covers premature termination, where the physical UI task is NOT done but the agent abruptly outputs a 'terminate' or 'answer' action, as well as over-execution, where the agent fails to recognize the overall goal is complete and hallucinates redundant UI actions instead of terminating.",
    "**Constraint Neglect:** The user instruction explicitly dictates multifaceted requirements. The agent executes an action that satisfies a portion of the instruction but explicitly disregards one or more of the specific constraints stated in the text. This includes positional negligence (e.g., the prompt says 'append to top' but the agent inserts at cursor) or explicit attribute negligence (e.g., the prompt says 'click the post by Bob' but the agent clicks a post by Alice). If the attribute wasn't explicitly requested in the text, ignoring it is not Constraint Neglect.",
    "**Action Formulation Error:** The agent attempts the EXACT SAME semantic intent but fails in formulating the action string. The intent must remain perfectly aligned with the 'Chosen Thought' and 'Chosen Proposed Step'. This manifests either as a syntax crash from a malformed string that breaks the parser (e.g., missing or malformed wrapper tags like omitting the opening `<tool_call>`, missing quotes like `{{\"action\": \"terminate\", \"status\": success}}` instead of `{{\"action\": \"terminate\", \"status\": \"success\"}}`, trailing commas, mismatched braces like `{{\"action\": \"type\", \"text\": \"hello\")` or as an invalid argument, such as passing merged values, omitting required JSON keys (e.g., `{{\"action\": \"left_click\"}}` missing the coordinate array), or using a non-existent enum action string like `{{\"action\": \"press_button\"}}`.",
    "**Observation Neglect:** The agent attempts to search, scroll, navigate, or open menus to find information or targets that are *already clearly visible* in the current observation.",
    "**Suboptimal Path:** Progresses the task but selects a highly inefficient, repetitive sequence of micro-actions when a faster standard UI paradigm is available. This includes input inefficiencies (e.g., relying on incremental UI controls like arrows or +/- buttons for large value changes instead of direct keyboard entry, or issuing repetitive 'backspace' actions to clear a text field instead of a bulk delete) and navigation inefficiencies (e.g., manually clicking through directories when a search bar is present).",
    "**Visual Hallucination:** The agent attempts to interact with a UI element, icon, or menu that strictly does not exist in the current visual observation. The agent fails to ground its action in the current screen pixels and instead relies on false internal priors. This manifests either as a temporal ghost interaction, where the agent attempts to interact with an element that existed in a previous state (trajectory history) but has since disappeared or been closed (e.g., clicking the 'X' of a modal that is no longer on screen), or as a prior-bias hallucination, where the agent blindly interacts with a coordinate based on standard UI layouts (e.g., assuming a 'Search' bar or 'Settings' gear is in the top right corner) without visually confirming it actually exists in the current UI rendering.",
    "**Timing and Latency Neglect:** The agent executes a logically correct subsequent action, but does so prematurely by failing to account for asynchronous UI changes. It completely ignores visual indicators that the system is busy (e.g., loading spinners, progress bars, disabled 'processing' buttons) or mid-transition (e.g., unfolding menus, sliding panels).",
    "**Memory & State Tracking Error:** The agent fails to properly utilize the trajectory history. It either erroneously repeats the exact same action it just executed (failing to realize a silent failure or success), or it inputs incorrect information because it failed to retain data gathered in an earlier step.",

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

def extract_tool_call(action_str: str):
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(action_str)
    return matches[0].strip() if matches else None


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



def add_trajectory_and_total_steps_df(df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy to avoid modifying your original dataframe directly
    df = df.copy()
    
    # 1. Identify distinct trajectories
    # Every time step_index is 1, it evaluates to True. 
    # cumsum() adds these up, creating a unique ID for each trajectory group.
    df['trajectory_id'] = (df['step_index'] == 1).cumsum()
    
    # 2. Add total_steps
    # Group by the new ID and broadcast the maximum step_index to all rows in that group
    df['total_steps'] = df.groupby('trajectory_id')['step_index'].transform('max')
    
    # 3. Add trajectory history
    def build_history(group):
        # Convert just the required columns for this specific trajectory into a list of dicts
        records = group[['operation', 'action']].to_dict(orient='records')
        
        # For row i, the previous history is everything from index 0 up to i (records[:i])
        history_list = [records[:i] for i in range(len(records))]
        
        # Return as a Pandas Series aligned with the group's original index
        return pd.Series(history_list, index=group.index)
        
    df['trajectory'] = df.groupby('trajectory_id', group_keys=False).apply(build_history)
    
    # Drop the temporary grouping column to keep your dataframe clean
    df = df.drop(columns=['trajectory_id'])
    
    return df


def is_valid_action_string(action_str: str, platform: str) -> bool:
    """
    Checks if an old action string is syntactically valid and contains
    only supported functions and non-None coordinates.
    """
    if pd.isna(action_str) or not str(action_str).strip():
        return False

    try:
        tree = ast.parse(str(action_str).strip(), mode="exec")
    except Exception:
        return False  # Catch malformed python strings

    if not tree.body:
        return False

    # Define the strict whitelists of supported old functions
    valid_desktop = {
        "click", "left_click", "doubleClick", "tripleClick", "rightClick", 
        "moveTo", "dragTo", "scroll", "swipe", "write", "type", "press", 
        "hotkey", "keyDown", "keyUp", "wait", "response", "answer", "terminate"
    }
    
    valid_mobile = {
        "click", "long_press", "swipe", "open_app", "navigate_home", 
        "navigate_back", "write", "type", "press", "key", "hotkey", 
        "wait", "response", "answer", "terminate"
    }

    allowed_funcs = valid_desktop if platform in ["desktop", "web", "windows", "mac", "ubuntu"] else valid_mobile

    for node in tree.body:
        # Reject if it's not a function call
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            return False

        call = node.value
        func_name = call.func.id if isinstance(call.func, ast.Name) else ""

        # Reject unsupported functions (like 'success' or 'failure')
        if func_name not in allowed_funcs:
            return False

        # Check the parsed kwargs for None values
        kwargs = {}
        for kw in call.keywords:
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                pass
        
        # Reject if x or y are explicitly set to None (prevents int(None) crashes)
        if "x" in kwargs and kwargs["x"] is None:
            return False
        if "y" in kwargs and kwargs["y"] is None:
            return False

    return True


def actions_are_identical(chosen_action_str, rejected_step_str):
    """
    Extracts the JSON from the <tool_call> blocks of both strings and compares them as dictionaries.
    Returns True if they execute the exact same tool call, False if they diverge.
    """
    def extract_dict(text):
        if not text: return None
        # Find everything between the tool_call tags (or just grab the first JSON block)
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        return None

    chosen_dict = extract_dict(chosen_action_str)
    rejected_dict = extract_dict(rejected_step_str)

    # If either failed to parse, we assume they are structurally different (or one is broken)
    if not chosen_dict or not rejected_dict:
        return False
        
    return chosen_dict == rejected_dict


def extract_think(gpt_response):
    """Extracts the text inside <think> tags."""
    match = re.search(r"<think>\s*(.*?)\s*</think>", gpt_response, re.DOTALL)
    return match.group(1).strip() if match else "No think specified."

def extract_operation(gpt_response):
    """Extracts the text inside <operation> tags."""
    match = re.search(r"<operation>\s*(.*?)\s*</operation>", gpt_response, re.DOTALL)
    return match.group(1).strip() if match else "No operation specified."

def extract_action(gpt_response):
    """Extracts the text inside <operation> tags."""
    match = re.search(r"<action>\s*(.*?)\s*</action>", gpt_response, re.DOTALL)
    return match.group(1).strip() if match else "No action specified."


def extract_all_tool_calls(action_str: str):
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in pattern.findall(action_str)]

def are_all_tool_calls_valid(action_str: str) -> bool:
    blocks = extract_all_tool_calls(action_str)
    
    if not blocks:
        return False
    
    for block in blocks:
        if not is_strict_tool_call(block):
            return False
    
    return True

def is_strict_tool_call(block: str) -> bool:
    block = block.strip()

    # Must start and end like a JSON dict
    if not (block.startswith("{") and block.endswith("}")):
        return False

    try:
        parsed = ast.literal_eval(block)
    except Exception:
        return False

    return isinstance(parsed, dict)


def extract_trajectories(trajectory_steps, image_head, env):
    """
    Iterates through a trajectory, accumulating history.
    When a terminal action is detected, it packages the current state 
    and accumulated trajectory for critic synthesis.
    """
    terminal_states = []
    history = []
    
    platform = "mobile" if env == "android" else "desktop"

    for i, step in enumerate(trajectory_steps):
        raw_action = step.get("action", "")
        operation = step.get("operation", "")
        think = step.get("think", "")
        
        # Map the action to the standardized agent format immediately
        mapped_action = map_old_action_to_agent_action(raw_action, operation, platform)

        image_path = os.path.join(image_head, step.get("image_path", ""))
        
        # Package it exactly how generate_negative_sample expects it
        terminal_states.append({
            "step_index": i + 1,
            "env": env,
            "platform": platform,
            "current_image_path": image_path,
            "task": step.get("task", ""),
            "think": think,
            "operation": operation,
            "action": mapped_action,
            "trajectory": list(history) # Deep copy the history up to this step
        })
            
        # Append the current step to the history for the NEXT steps
        history.append({
            "operation": operation,
            "action": mapped_action
        })
        
    return terminal_states


def load_and_group_trajectories(jsonl_file_path):
    trajectories = defaultdict(list)
    
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            image_path = data.get("image", "")
            
            # Extract the trajectory ID (e.g., "9f4l7axy/oeyzgchn")
            path_parts = image_path.split('/')
            if len(path_parts) >= 2:
                traj_id = f"{path_parts[0]}/{path_parts[1]}"
            else:
                continue # Skip invalid paths
                
            # Extract the operation from the 'gpt' conversation turn
            think_text = ""
            operation_text = ""
            action_text = ""
            for msg in data.get("conversations", []):
                if msg.get("from") == "gpt":
                    think_text = extract_think(msg.get("value", ""))
                    operation_text = extract_operation(msg.get("value", ""))
                    action_text = extract_action(msg.get("value", ""))

                    break
                
            task = re.search(r"Task:\s*(.*)", data["conversations"][0]["value"]).group(1)
            # Append step to the specific trajectory
            trajectories[traj_id].append({
                "image_path": image_path,
                "task": task,
                "think": think_text,
                "operation": operation_text,
                "action": action_text,
            })
            
    return trajectories



def generate_negative_sample(row, client, n=3):
    """
    Takes a single row from the DataFrame and generates the reasoning
    """
    # 1. Randomly sample 'n' dimensions and inject them into the system prompt
    if row['env'] == 'android':
        platform = 'mobile'
    elif row['env'] in ['web', 'ubuntu', 'windows', 'mac']:
        platform = 'desktop'
    else:
        ValueError(f"{row['env']} is not defined.")

    # Attach current observation
    image_path = row['image_path']
    base64_image = encode_image_to_base64(image_path)

    task = row['task']
    chosen_thought = row['think']
    chosen_operation = row['operation']
    chosen_action = row['action']

    trajectory_list = row['trajectory']

    # 2. Format the Trajectory list into a readable string for the prompt
    trajectory_str = ""
    for i, step in enumerate(trajectory_list):
        trajectory_str += (
            f"Step {i+1}:\n"
            f"Action: {step.get('operation')}\n"
            f"{step.get('action')}\n\n"
        )
    trajectory_str = trajectory_str.strip()

    # Format the User Message
    user_prompt = f"User Instruction (Goal):\n{task}\n\n"
    user_prompt += f"Trajectory:\n{trajectory_str}\n\n"
    user_prompt += f"Current Observation:\n<image>\n\n"
    user_prompt += f"Chosen Thought:\n{chosen_thought}\n\n"
    user_prompt += f"Chosen Proposed Step:\nAction: {chosen_operation}\n{chosen_action}\n\n"

    text_before, text_after = user_prompt.split("<image>")

    # --- CONDITIONAL ROUTING FOR HARD NEGATIVES ---
    # Find the specific hard-negative dimensions dynamically
    term_dim = next((d for d in error_dimensions if "Termination Misjudgment" in d), None)
    latency_dim = next((d for d in error_dimensions if "Timing and Latency Neglect" in d), None)
    vector_dim = next((d for d in error_dimensions if "Parameter Vector Miscalibration" in d), None)

    selected_dims = []

    # ANTI-BIAS: 80% chance to force the hard negative, 20% chance to be completely random
    force_hard_negative = random.random() < 0.80
    
    if force_hard_negative:
        # Route 1: Task Finish/Reporting -> Force Termination Misjudgment
        if '"action": "answer"' in chosen_action and term_dim:
            selected_dims.append(term_dim)
            pool = [d for d in error_dimensions if d != term_dim]
            selected_dims.extend(random.sample(pool, k=n-1))

        elif '"action": "terminate"' in chosen_action and term_dim:
            selected_dims.append(term_dim)
            pool = [d for d in error_dimensions if d != term_dim]
            selected_dims.extend(random.sample(pool, k=n-1))

        # Route 2: Direction Issue
        elif ('"action": "scroll"' in chosen_action or '"action": "hscroll"' in chosen_action or '"action": "swipe"' in chosen_action or '"action": "left_click_drag"' in chosen_action) and vector_dim:
            selected_dims.append(vector_dim)
            pool = [d for d in error_dimensions if d != vector_dim]
            selected_dims.extend(random.sample(pool, k=n-1))

        # Route 3: Passive Waiting -> Force Timing and Latency Neglect
        elif '"action": "wait"' in chosen_action and latency_dim:
            selected_dims.append(latency_dim)
            pool = [d for d in error_dimensions if d != latency_dim]
            selected_dims.extend(random.sample(pool, k=n-1))  
        # Fallback if it didn't match the specific routes
        else:
            selected_dims = random.sample(error_dimensions, k=n)
    else:
        selected_dims = random.sample(error_dimensions, k=n)

    # ANTI-BIAS: Shuffle the list to destroy Primacy Bias (Index 0 focus)
    random.shuffle(selected_dims)

    dimensions_str = "\n\n".join(selected_dims)
    # ----------------------------------------------

    system_prompt = negative_sample_generation_prompt.format(
        three_dimensions=dimensions_str, 
        platform_action_space=action_spaces[platform]
    )

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
                        {"type": "text", "text": text_before.strip()},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": text_after.strip()},

                    ]}
                ],
            )
            # Extract the reasoning and the structured variables
            reasoning_text = response.choices[0].message.reasoning.strip()
            parsed_json = extract_json(response.choices[0].message.content.strip())

            if not parsed_json:
                print(f"[Attempt {attempt}] Failed to parse JSON format.")
                continue

            # 4. Validation Step: Check if it generated the 'synthesis' array successfully
            dimension_violated = str(parsed_json.get("dimension_violated", "")).strip()
            rejected_thought = str(parsed_json.get("rejected_thought", "")).strip()
            rejected_proposed_step = str(parsed_json.get("rejected_proposed_step", "")).strip()

            if not are_all_tool_calls_valid(rejected_proposed_step):
                print(f"[Attempt {attempt}] Invalid action JSON format.")
                continue

            # Validation Step
            if dimension_violated and rejected_thought and rejected_proposed_step:
                # Check for absolute divergence
                if not actions_are_identical(chosen_action, rejected_proposed_step):
                    if "<tool_call>" in rejected_proposed_step:
                        final_json = parsed_json
                        break
                    else:
                        print(f"[Attempt {attempt}] Missing <tool_call> tags in proposed step.")
                else:
                    print(f"[Attempt {attempt}] Divergence check failed: Generated action is identical to chosen action.")
            else:
                print(f"[Attempt {attempt}] Output mismatch: Missing required JSON keys.")
                
        except Exception as e:
            print(f"[Attempt {attempt}] Error generating reasoning for step {row.get('step_index')}: {e}")

    return final_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_transition_annotation", default=None)
    parser.add_argument("--save_dir", default="./output")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'negative_samples.jsonl')
    client_vllm = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    random.seed(44)

    # Extract trajectory
    json_files = glob.glob("./ScaleCUA-Data/refined_annotations/*.jsonl")
    all_samples = []

    for jsonl_path in tqdm(json_files, total=len(json_files)):
        grouped_trajectories = load_and_group_trajectories(jsonl_path)

        file_heading = os.path.basename(jsonl_path).split("_")

        if len(file_heading) >= 3:
            dataset_name = f"{file_heading[0]}_{file_heading[1]}"
            environment = file_heading[2].split(".")[0]
        else:
            continue # Skip malformed filenames
            
        image_heading = f"./ScaleCUA-Data/data/{dataset_name}/{environment}/images"

        for traj_id, steps in grouped_trajectories.items():
            terminal_states = extract_trajectories(steps, image_heading, environment)
            all_samples.extend(terminal_states)

    df_trajectory = pd.DataFrame(all_samples)

    # 1. Load the single JSONL file into a pandas DataFrame
    df = pd.read_json(args.state_transition_annotation, lines=True)

    df = df.rename(columns={"current_image_path": "image_path", "current_action": "action", "current_operation": "operation", "current_think": "think", "environment": "env"}).drop(columns=['next_image_path', 'action_type'])
    df['platform'] = df['env'].apply(
            lambda x: "mobile" if x == "android" else "desktop"
        )

    # 2. FILTER: Remove invalid action strings (success(), None coordinates, etc.)
    valid_mask = df.apply(
        lambda row: is_valid_action_string(row.get('action', ''), row['platform']), 
        axis=1
    )
    df = df[valid_mask].copy()


    df['action'] = df.apply(
            lambda row: map_old_action_to_agent_action(row.get('action', ''), row.get('operation', ''), row['platform']),
            axis=1
        )

    df = df.merge(
        df_trajectory[['current_image_path', 'trajectory']],
        left_on="image_path",
        right_on="current_image_path",
        how="inner"
    )
    df = df.drop(columns=['current_image_path'], errors='ignore')

    # total_count = len(df)
    # df = df[:int(total_count * 0.5)]
    df = df.sample(frac=1).reset_index(drop=True)

    # 3. Generate reasoning for each row
    negative_samples = [None] * len(df) # Pre-allocate list to maintain correct order
    
    # Wrapper function for the threads to keep track of the original row index
    def process_row(index, row):
        try:
            result = generate_negative_sample(row, client_vllm)
            return index, result
        except Exception as e:
            print(f"Unhandled error on row {index}: {e}")
            return index, None

    # Use ThreadPoolExecutor to send multiple requests to vLLM at the same time
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        futures = [executor.submit(process_row, index, row) for index, row in df.iterrows()]
        
        # Process them as they complete, wrapped in tqdm for a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df)):
            idx, result = future.result()
            negative_samples[idx] = result # Place result back in the correct original order

    dimensions_violated = []
    rejected_thoughts = []
    rejected_operations = []
    rejected_proposed_steps = []

    for res in negative_samples:
        if res is not None:
            clean_dimension = res.get("dimension_violated", "").replace("**", "").strip()
            dimensions_violated.append(clean_dimension)
            rejected_thoughts.append(res.get("rejected_thought", ""))
            rejected_proposed_steps.append(res.get("rejected_proposed_step", "")) 
        else:
            # If the API call failed, append None so we can filter it out easily
            dimensions_violated.append(None)
            rejected_thoughts.append(None)
            rejected_proposed_steps.append(None)

    # 5. Add the 4 new columns to the DataFrame
    df['dimension_violated'] = dimensions_violated
    df['rejected_thought'] = rejected_thoughts
    df['rejected_proposed_step'] = rejected_proposed_steps

    # 6. Filter out rows where the generation failed (where values are None)
    original_len = len(df)
    df_filtered = df[df['dimension_violated'].notnull()]
    dropped_count = original_len - len(df_filtered)

    print(f"\n--- Filtering Summary ---")
    print(f"Original samples: {original_len}")
    print(f"Failed extractions dropped: {dropped_count}")
    print(f"Final valid samples: {len(df_filtered)}\n")

    # 7. Save the updated DataFrame back to JSONL
    df_filtered.to_json(
        save_path,
        orient='records',
        lines=True,
        force_ascii=False
    )

    print("Pipeline completed successfully.")