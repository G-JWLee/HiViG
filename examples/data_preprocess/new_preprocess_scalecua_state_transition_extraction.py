import argparse
import ast
import base64
import glob
import hashlib
import io
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# -------------------------------------------------------------------------
# 2. PROMPT TEMPLATE
# -------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert UI State Transition Analyzer. Your job is strictly to evaluate environmental dynamics. DO NOT plan future actions or solve the task.

### Task Description
You are given:
- User Instruction (Goal): The overall, high-level task the user wants the agent to accomplish.
- Current Observation: The screenshot visual representation of the current screen.
- Current Proposed Step: The combined verbalized intent (Action: ...) and the executable command (<tool_call>...</tool_call>) suggested by the agent for the current observation.
- Next State Observation: The screenshot visual representation of the screen as a result of the given current action.

Your sole objective is to explain the visual changes that occurred between two screenshots after a specific action was executed in GUI environments. To be successful, it is very important to understand the causal effect of the current action on the next state of the screen.

## Action Space
{platform_action_space}

### Output Format & Constraints:
Follow these strict rules for reasoning on the state prediction:

1. **Forward-Predictive Tense (Crucial):** You MUST write your `<next_state>` block using the future tense, as if you are a physics engine predicting what will happen the moment the action is executed. 
   - DO NOT USE PAST OR PRESENT TENSE for the result: "The menu opened", "The screen shows..."
   - USE NATURAL FUTURE TENSE: "The menu will open", "A modal is expected to appear", "The layout will shift..."
   - AVOID REPETITIVE BOILERPLATE: Do not start every prediction with the exact same phrase. Vary your sentence structures naturally while strictly maintaining the forward-looking, objective future tense.
2. **Focus on the UI Skeleton (CRITICAL):** Describe ONLY the structural UI changes triggered by the action (e.g., the containers, layouts, modals, and widgets). Do NOT read or transcribe the data populating those structures. We only care about the environmental dynamics, not the specific content.
   - CORRECT FOCUS: "A grid layout will appear," "A bottom navigation bar will load," "The screen will transition to a detailed view."
   - INCORRECT FOCUS (DO NOT DO THIS): Mentioning specific book titles, exact percentages, usernames, clock times, or specific search results.
3. **Strictly Objective Tone:** Do NOT use first-person pronouns ("I", "me", "we"). You are an objective system observer.
4. **Structured Reasoning:** Your `<think>` block must follow this 2-step causal breakdown:
   - [Action Mechanism]: What UI element was interacted with in the Current Observation?
   - [Structural Effect]: What is the layout/skeleton of the resulting state, ignoring all dynamic content?
5. **Final Output:** Output your structural, forward-looking state prediction inside a <next_state> block based strictly on the [Structural Effect].

<think>
Your reasoning here...
</think>
<next_state>
Your final state prediction here...
</next_state>
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
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found.")
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

def get_file_hash(filepath):
    """Computes the MD5 hash of a file for fast identical-check comparison."""
    if not os.path.exists(filepath):
        return None
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# ---------------------------------------------------------
# 2. Parse and Group Trajectories
# ---------------------------------------------------------

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


def create_state_transition_pairs(trajectory_steps, image_head, env, filtered_log):
    """
    Creates (Observation t, Action t, Observation t+1) pairs from a list of trajectory steps.
    Filters out pairs where the current and next images are perfectly identical.
    """
    transition_pairs = []
    
    # We need at least 2 steps to form a valid transition
    if len(trajectory_steps) < 2:
        return transition_pairs

    # Loop until the second-to-last step
    for i in range(len(trajectory_steps) - 1):
        current_step = trajectory_steps[i]
        next_step = trajectory_steps[i + 1]
        
        current_image_path = os.path.join(image_head, current_step.get("image_path", ""))
        next_image_path = os.path.join(image_head, next_step.get("image_path", ""))

        # ---------------------------------

        actions = current_step.get("action", "")
        actions = actions.split("\n")
        methods = []
        for action in actions:
            func_match = re.match(r"(\w+)\((.*)\)$", action.strip())
            if func_match:
                method = func_match.group(1)
                methods.append(method)
        method_name = "\n".join(methods)

        # 2. Check if the action contains a terminal/response command
        terminal_methods = {"response", "answer", "terminate"}
        is_terminal_action = any(m in terminal_methods for m in methods)

        # --- NEW: Image Identity Check ---
        hash_current = get_file_hash(current_image_path)
        hash_next = get_file_hash(next_image_path)

        if hash_current and hash_next and hash_current == hash_next and not is_terminal_action:
            # Log the identical images to filter out later
            filtered_log.append({
                "task": current_step.get("task", ""),
                "step_index": i + 1,
                "current_image_path": current_image_path,
                "next_image_path": next_image_path,
                "action_executed": current_step.get("action", "")
            })
            continue # Skip adding to transition_pairs

        if env == 'ubuntu':
            continue

        transition_pair = {
            "step_index": i + 1,
            "task": current_step.get("task", ""),
            "current_image_path": current_image_path,
            "current_action": current_step.get("action", ""),
            "current_operation": current_step.get("operation", ""), 
            "current_think": current_step.get("think", ""),        
            "next_image_path": next_image_path,
            "action_type": method_name,
            "environment": env,
        }
        
        transition_pairs.append(transition_pair)
        
    return transition_pairs


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


def get_verbalized_transition(client, step_pair):
    # URL is often not explicitly in the snippet provided, but usually in metadata.
    # We will use a placeholder or extract if available in your full json.
    intent = step_pair["task"]
    current_operation = step_pair["current_operation"]
    current_action = step_pair["current_action"]
    base64_current_observation = encode_image_to_base64(step_pair['current_image_path'])
    base64_next_observation = encode_image_to_base64(step_pair["next_image_path"])

    if step_pair['environment'] == 'android':
        platform = 'mobile'
    elif step_pair['environment'] in ['web', 'ubuntu', 'windows', 'mac']:
        platform = 'desktop'
    else:
        ValueError(f"{step_pair['environment']} is not defined.")

    proposed_step = map_old_action_to_agent_action(current_action, current_operation, platform)


    if not base64_current_observation or not base64_next_observation:
        return None
    
    system_prompt = SYSTEM_PROMPT.format(platform_action_space=action_spaces[platform])
    user_prompt = f"User Instruction (Goal):\n{intent}\n\nCurrent Proposed Step:\n{proposed_step}\n"

    model_name = "Qwen/Qwen3-VL-32B-Thinking"
    attempt = 0

    rationale = None
    state_transition = None
    while attempt < 5:
        attempt += 1
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "text", "text": "Current Observation:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_current_observation}"}},
                    {"type": "text", "text": "Next State Observation:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_next_observation}"}},
                ]}
            ]

            response = client.chat.completions.create(# type: ignore
                model=model_name,
                messages=messages,
            )

            rationale = response.choices[0].message.reasoning.strip()
            state_transition = response.choices[0].message.content.strip()

            match = re.search(r"<next_state>\s*(.*?)\s*</next_state>", state_transition, re.DOTALL)
            state_transition = match.group(1).strip() if match else state_transition

            if rationale is not None:
                break

        except Exception as e:
            print(f"[Attempt {attempt}] Error generating rationale: {e}")

    return rationale, state_transition

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    args = parser.parse_args()
    
    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)

    client_vllm = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    json_files = glob.glob("./ScaleCUA-Data/refined_annotations/*.jsonl")
    state_transition_pairs = []
    
    # List to hold identical image pair data
    identical_images_log = []

    for jsonl_path in tqdm(json_files, total=len(json_files)):
        # 1. Group the trajectories
        grouped_trajectories = load_and_group_trajectories(jsonl_path)

        image_heading = os.path.basename(jsonl_path)
        image_heading = image_heading.split("_")
        environment = image_heading[2]

        image_heading = os.path.join(f"/ScaleCUA-Data/data/{image_heading[0]}_{image_heading[1]}/{image_heading[2]}/images")

        for traj_id, steps in grouped_trajectories.items():
            # Pass the identical_images_log list to collect skipped pairs
            pairs = create_state_transition_pairs(steps, image_heading, environment, identical_images_log)
            state_transition_pairs.extend(pairs)
    
    # Save the log of identical images to a file
    log_save_path = os.path.join(local_save_dir, "identical_images_log.jsonl")
    with open(log_save_path, 'w') as f:
        for log_item in identical_images_log:
            f.write(json.dumps(log_item) + "\n")
    print(f"Filtered out {len(identical_images_log)} identical image pairs. Log saved to {log_save_path}")

    annotated_samples = []

    # Helper function to unpack arguments and return the modified pair
    def process_pair(pair):
        rationale, state_transition = get_verbalized_transition(client_vllm, pair)
        if state_transition is not None:
            pair.update({
                "state_transition": state_transition,
                "transition_rationale": rationale
            })
            return pair
        return None

    # Set concurrency level. Start with 16-32 depending on your vLLM server's capacity.
    MAX_WORKERS = 20 

    print(f"Running vLLM inference in parallel with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the thread pool
        futures = {executor.submit(process_pair, pair): pair for pair in state_transition_pairs}
        
        # Process results as they complete (order doesn't matter here)
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result is not None:
                    annotated_samples.append(result)
            except Exception as e:
                print(f"Task generated an exception: {e}")

    df = pd.DataFrame(annotated_samples)

    save_path = os.path.join(local_save_dir, "state_transition_annotation.jsonl")

    df.to_json(
        save_path,
        orient='records',
        lines=True,
        force_ascii=False
    )

    print("done")