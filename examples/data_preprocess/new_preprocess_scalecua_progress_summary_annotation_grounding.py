import argparse
import ast
import base64
import glob
import hashlib
import io
import json
import math
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from PIL import Image, ImageDraw
from tqdm import tqdm

# -------------------------------------------------------------------------
# 2. PROMPT TEMPLATE
# -------------------------------------------------------------------------

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

- Perform a fast visual diff. State the 'Previous Proposed Step', compare the 'Previous Observation' to the 'Current Observation', and explicitly describe exactly what changed physically on the screen. 
- Definitively conclude whether the 'Previous Proposed Step' resulted in a UI update or a silent failure.
- You must describe the UI state exactly as rendered. Do NOT diagnose errors when describing UI states. UI elements frequently truncate, scroll, or obscure text. Simply report the visible characters without judging the agent's accuracy.
- Map the verified 'Current Observation' to the 'User Instruction (Goal)'. Identify the visual gap: what goal-related UI elements are currently present, and what are missing? You are a reflective summarizer, NOT a planner. You are STRICTLY FORBIDDEN from predicting, suggesting, or mentioning what the "next step" should be. Only evaluate the current static reality.
</think>
<summary>
Progress Summary: [LONG TERM MEMORY: Synthesize the 'Previous Summary' with the current result. Compress completed micro-steps into dense macro-achievements (e.g., "Successfully logged in"). Only append the new action if it succeeded. When noting missing goal constraints, use strictly descriptive, state-based language rather than prescriptive instructions.]
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



def load_and_group_trajectories(jsonl_file_path):
    trajectories = defaultdict(list)
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            image_path = data.get("image", "")
            
            path_parts = image_path.split('/')
            if len(path_parts) >= 2:
                traj_id = f"{path_parts[0]}/{path_parts[1]}"
            else:
                continue 
                
            think_text, operation_text, action_text = "", "", ""
            for msg in data.get("conversations", []):
                if msg.get("from") == "gpt":
                    value = msg.get("value", "")
                    think_text = extract_think(value)
                    operation_text = extract_operation(value)
                    action_text = extract_action(value)
                    break
                
            task = re.search(r"Task:\s*(.*)", data["conversations"][0]["value"]).group(1)
            
            trajectories[traj_id].append({
                "image_path": image_path,
                "task": task,
                "think": think_text,
                "operation": operation_text,
                "action": action_text,
            })
    return trajectories


# -------------------------------------------------------------------------
# 3. SEQUENTIAL INFERENCE
# -------------------------------------------------------------------------

def get_sequential_summary(client, intent, prev_summary, prev_img_b64, current_img_b64, prev_proposed_step, platform):
    """Calls the LLM to get the updated summary based on the visual diff."""
    system_prompt = SUMMARY_SYSTEM_PROMPT.format(platform_action_space=action_spaces[platform])
    
    user_prompt = SUMMARY_USER_PROMPT.format(intent=intent, previous_summary=prev_summary, proposed_step=prev_proposed_step)

    text_before, text_middle, text_after = user_prompt.split("<image>")

    model_name = "Qwen/Qwen3-VL-32B-Thinking"
    attempt = 0

    while attempt < 5:
        attempt += 1
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": text_before.strip()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prev_img_b64}"}},
                    {"type": "text", "text": text_middle.strip()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_img_b64}"}},
                    {"type": "text", "text": text_after.strip()},
                ]}
            ]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )

            rationale = response.choices[0].message.reasoning.strip()
            full_response = response.choices[0].message.content.strip()

            # Extract the raw summary block to pass to the next step
            match = re.search(r"<summary>\s*(.*?)\s*</summary>", full_response, re.DOTALL)
            clean_summary = match.group(1).strip() if match else full_response

            return rationale, clean_summary

        except Exception as e:
            print(f"[Attempt {attempt}] Error generating summary: {e}")

    return None, None

def process_single_trajectory(traj_id, steps, image_heading, env, client):
    """
    Chronologically processes a single trajectory, passing the summary state forward.
    Returns a list of annotated steps.
    """
    annotated_steps = []
    
    # Initialize the starting memory state
    rolling_summary = "No actions taken yet. This is the beginning of the task."
    
    platform = 'mobile' if env == 'android' else 'desktop'

    for i in range(len(steps) - 1):
        prev_step = steps[i]
        curr_step = steps[i + 1]

        # Format the verbalized intent and tool call from the previous step
        proposed_action_str = map_old_action_to_agent_action(
            prev_step["action"], prev_step["operation"], platform
        )

        if random.random() < 0.5:
            proposed_step_formatted = f"{proposed_action_str}"
        else:
            proposed_step_formatted = f"Action: {prev_step['operation']}\n{proposed_action_str}"

        prev_img_path = os.path.join(image_heading, prev_step.get("image_path", ""))
        curr_img_path = os.path.join(image_heading, curr_step.get("image_path", ""))

        prev_img_b64 = encode_image_to_base64(prev_img_path)
        curr_img_b64 = encode_image_to_base64(curr_img_path)

        if not prev_img_b64 or not curr_img_b64:
            continue

        # prev_img_b64 = mark_action_on_base64_image(proposed_action_str, prev_img_b64)

        # Get the new summary based on the visual diff
        rationale, new_summary = get_sequential_summary(
            client=client,
            intent=prev_step["task"],
            prev_summary=rolling_summary,
            prev_img_b64=prev_img_b64,
            current_img_b64=curr_img_b64,
            prev_proposed_step=proposed_step_formatted,
            platform=platform
        )

        if new_summary:
            # Save the annotated pair
            annotated_steps.append({
                "trajectory_id": traj_id,
                "step_index": i + 1,
                "task": prev_step["task"],
                "prev_image_path": prev_img_path,
                "current_image_path": curr_img_path,
                "proposed_step": proposed_step_formatted,
                "summary_rationale": rationale,
                "previous_summary": rolling_summary,
                "generated_summary": new_summary,
                "environment": env
            })
            
            # Update the rolling state for the next iteration (i+1)
            rolling_summary = new_summary

    return annotated_steps



# -------------------------------------------------------------------------
# 4. MAIN PIPELINE
# -------------------------------------------------------------------------

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
    all_annotated_samples = []

    print("Grouping trajectories from JSONL files...")
    trajectory_jobs = []

    for jsonl_path in tqdm(json_files, desc="Parsing Files"):
        grouped_trajectories = load_and_group_trajectories(jsonl_path)

        image_heading_parts = os.path.basename(jsonl_path).split("_")
        environment = image_heading_parts[2]
        if environment == 'ubuntu':
            continue

        image_heading = os.path.join(f"/ScaleCUA-Data/data/{image_heading_parts[0]}_{image_heading_parts[1]}/{environment}/images")

        for traj_id, steps in grouped_trajectories.items():
            if len(steps) >= 2:
                trajectory_jobs.append((traj_id, steps, image_heading, environment))

    # We parallelize across trajectories, NOT individual steps
    MAX_WORKERS = 16
    print(f"Running vLLM inference across {len(trajectory_jobs)} trajectories using {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each full trajectory to the thread pool
        futures = {
            executor.submit(
                process_single_trajectory, job[0], job[1], job[2], job[3], client_vllm
            ): job for job in trajectory_jobs
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Trajectories"):
            try:
                trajectory_results = future.result()
                if trajectory_results:
                    all_annotated_samples.extend(trajectory_results)
            except Exception as e:
                print(f"Trajectory task generated an exception: {e}")

    df = pd.DataFrame(all_annotated_samples)
    save_path = os.path.join(local_save_dir, "sequential_summary_annotation.jsonl")

    df.to_json(
        save_path,
        orient='records',
        lines=True,
        force_ascii=False
    )

    print(f"Done! Saved {len(df)} sequential annotations to {save_path}")