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

rejected_action_rationale_system_prompt = """
You are an expert World Model and Action Critic for a GUI navigation agent.

Your task is to generate the intermediate reasoning steps and final output that a perfect Action Critic would generate naturally for a FLAWED action.

I will provide you with:
- User Instruction (Goal): The overall, high-level task the user wants the agent to accomplish.
- Trajectory: A sequential record of the agent's past proposed steps.
- Current Observation: The screenshot visual representation of the current screen. **Note: IF the agent's proposed step involves spatial coordinates, a visual marker (red 'X' or blue arrow) has been injected into this image. If the action is non-spatial (e.g., type, wait), the image is clean.**
- Current Proposed Step: The combined verbalized intent (Action: ...) and the executable command (<tool_call>...</tool_call>) suggested by the agent for the current observation.
- **Target Error Dimension**: The exact error dimension you must classify this action into.
- **Reference Proposed Step**: The actual agent step that should have been suggested. (Hidden context).
- **Reference Action Mechanism**: The UI element the reference proposed step interacted with.
- **Reference Structural Effect**: The physical layout change that would have happened when executing the reference proposed step.

You must write the exact output following this strict structure:

<think>
[Act as a seamless, organic internal monologue. You must naturally deduce the 'Overall Grading: Bad' and 'Target Error Dimension' by reasoning through the following aspects. You MUST separate your reasoning into three distinct paragraphs using double newlines (\n\n). DO NOT use explicit numbered lists, bullet points, or section headers.]

In your first paragraph, start exactly with this format: "Action String Check: [Insert the EXACT <tool_call> string here] - Syntax is [Valid/Invalid]."
**CRITICAL VISUAL GROUNDING:** Next, you MUST evaluate ALL `<tool_call>` blocks specifically inside the 'Current Proposed Step'. **CRITICAL WARNING: You must completely IGNORE the 'Reference Proposed Step' for this visual anchoring phase.**
- **If ANY tool call in the 'Current Proposed Step' contains `coordinate` arguments (e.g., click, swipe, mouse_move):** The coordinates perfectly match the injected visual marker. Locate the **red 'X' marker** or the **blue arrow**. To prevent spatial hallucinations, you must perform a strict, literal visual scan:
    1. **Literal Center:** First, read the exact text or identify the specific icon located directly beneath the exact center intersection of the marker.
    2. **Adjacency Check:** Identify the elements immediately adjacent (left/right/above/below) to the marker.
    3. **Final Identification:** Based strictly on the literal center, explicitly describe exactly what UI element (or empty space) is targeted. (e.g., "The X is directly on the 'Subscribe' text, adjacent to the '0' counter. Therefore, it targets the Subscribe button, not the counter.")
- **If NO tool calls in the 'Current Proposed Step' contain `coordinate` arguments (e.g., ONLY scroll, type, key, wait, terminate):** You are STRICTLY FORBIDDEN from looking for or mentioning a red 'X' or blue arrow. **CRITICAL WARNING: If you see a red 'X' in the image during a non-coordinate action, it is a residual artifact from a previous step. You MUST completely ignore it.** Evaluate the global screen state and the logical parameters provided in the 'Current Proposed Step' JSON.
Secretly compare this visual reality to the provided 'Reference Action Mechanism' to understand where the agent deviated.
Next, analyze the 'Trajectory' against the 'User Instruction (Goal)' to establish the current progress state.
Finally, conclude this paragraph by explicitly stating what UI element the action targets in the context of this current state. You MUST STRICTLY AVOID comparative phrasing like "it clicked X instead of Y" or judging the ultimate failure of the action here. Evaluate the flawed action's visual target objectively in a vacuum.

In your second paragraph, you MUST perform a deterministic consequence prediction to avoid hallucinating successful state transitions. Predict the exact physical consequence of the flawed action using forward-predictive future tense strictly based on the following assigned mechanic:
**Target Consequence Instruction:** {target_consequence_instruction}

In your third paragraph, compare your predicted consequence directly to the 'User Instruction (Goal)'. You must organically deduce why this predicted structural UI change (or lack thereof) is a failure.
**CRITICAL CONCEALMENT:** You possess the hidden 'Reference' variables, but the final critic model will NOT. Therefore, you must use the 'Reference' variables secretly in your own mind to understand the true UI physics rule. However, you MUST write your deduction as if you are discovering this rule PURELY by observing the flawed action and the visual screen state.
NEVER use the words 'reference', 'optimal', 'target', or 'should have'. Do not compare the flawed action to the reference action in text. Instead, state the environmental rule as an objective fact of the UI (e.g., "Because the visual marker is placed on an un-clickable div, the dropdown will not open, preventing the agent from progressing"). Conclude that the action is 'Bad', explicitly state the 'Target Error Dimension', and pinpoint the specific environmental rule violated.
</think>
<criticism>
Overall Grading: Bad
Error Dimension: [State Target Error Dimension here.]
Explanation: [Describe the immediate physical/structural consequence of the flawed action in the future tense. Then, state the universal UI rule the agent violated, and explain the correct environmental mechanic required for this type of interaction. Do NOT explicitly name the Error Dimension.]
</criticism>

## Action Space
{platform_action_space}

## TARGET ERROR DIMENSIONS
{error_dimension_str}

## CONSTRAINTS (STRICT):
1. NO META-LANGUAGE: Act entirely as an autonomous critic. You are STRICTLY FORBIDDEN from using the words "Ideal", "Target", "Optimal", "flawed", or phrases like "The provided label is...", or acknowledging that you were given hidden context variables. In your <think> block, frame your deduction purely as an observation of the UI (e.g., write "The dedicated dropdown arrow is the correct trigger..." instead of "The reference action is the dropdown arrow...").
2. IN-CHARACTER DEDUCTION: You must frame the diagnosis as your own organic deduction derived directly from the Current Observation (and the visual markers) and the Agent's inputs. Do not use robotic phrasing like 'The trajectory shows...'. Instead, deduce the current state naturally.
3. DESCRIPTIVE, NOT PRESCRIPTIVE: Inside <criticism>, you MUST NOT suggest specific commands, exact x/y coordinates, or exact python syntax to fix the error. You must only teach the environment (e.g., "The field must be focused first.")
4. OUTPUT FORMAT: Output ONLY the <think> and <criticism> blocks.
5. NO HEADERS: Your <think> block must be a continuous stream of consciousness. You are STRICTLY FORBIDDEN from printing headers like "Visual Anchoring" or "Paragraph 1". Write purely in natural paragraphs.
6. You must use the 'Reference' state variables only to reverse-engineer the UI physics rule. Inside your final <criticism>, you are STRICTLY FORBIDDEN from mentioning the reference action or telling the agent what it should have done. You must only explain why its current action physically failed.
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

error_dimensions = {
    "Grounding/Spatial Error": "The agent attempts the EXACT SAME semantic intent but fails in coordinate precision. It outputs coordinates that land in the immediate adjacent 'dead space'—close enough to look like a near-miss visually, but definitively OUTSIDE the interactive bounding box of the target. (e.g., the white margins between list items, the empty gutter of the screen, accidentally clipping the edge of an adjacent but incorrect UI element). Do not use a fixed pixel offset. You must visually identify the boundaries of the target's clickable area (thumbnail + text container) and ensure the new coordinate lands definitively in the non-interactive space just outside of those boundaries.",
    "Procedural Prerequisite Neglect": "Attempts a logical operation but skips a mandatory preceding interface state change. This includes logical neglect, such as skipping a required interaction rule (e.g., compressing files without selecting them first, typing without clicking to focus), as well as spatial or occlusion neglect, where the agent attempts to interact with a background element without first dismissing a foreground overlay (like a modal, pop-up, or sticky menu) blocking the target on the Z-axis.",
    "Semantic Error": "Perfectly targets a valid, clickable element, but fundamentally misunderstands the inherent meaning, function, or consequence of that specific UI element in relation to the goal. The error stems purely from misinterpreting vocabulary, icons, or standard UI paradigms (e.g., clicking 'Sign Up' when trying to log into an existing account, or clicking a deceptive ad disguised as a download button).",
    "Termination Misjudgment": "The agent fundamentally misunderstands the completion status of the User Instruction (Goal) based on the visual state. A critical manifestation of this is a reporting failure, where the agent successfully locates the required information but incorrectly outputs a 'terminate' action BEFORE using the 'answer' tool when the overall goal requires explicitly reporting the text back to the user. This dimension also covers premature termination, where the physical UI task is NOT done but the agent abruptly outputs a 'terminate' or 'answer' action, as well as over-execution, where the agent fails to recognize the overall goal is complete and hallucinates redundant UI actions instead of terminating.",
    "Action Formulation Error": "The agent attempts the EXACT SAME semantic intent but fails in formulating the action string. The intent must remain perfectly aligned with the 'Chosen Thought' and 'Chosen Proposed Step'. This manifests either as a syntax crash from a malformed string that breaks the parser (e.g., missing or malformed wrapper tags like omitting the opening `<tool_call>`, missing quotes like `{{\"action\": \"terminate\", \"status\": success}}` instead of `{{\"action\": \"terminate\", \"status\": \"success\"}}`, trailing commas, mismatched braces like `{{\"action\": \"type\", \"text\": \"hello\")` or as an invalid argument, such as passing merged values, omitting required JSON keys (e.g., `{{\"action\": \"left_click\"}}` missing the coordinate array), or using a non-existent enum action string like `{{\"action\": \"press_button\"}}`.",
    "Observation Neglect": "The agent attempts to search, scroll, navigate, or open menus to find information or targets that are *already clearly visible* in the current observation.",
    "Suboptimal Path": "Progresses the task but selects a highly inefficient, repetitive sequence of micro-actions when a faster standard UI paradigm is available. This includes input inefficiencies (e.g., relying on incremental UI controls like arrows or +/- buttons for large value changes instead of direct keyboard entry, or issuing repetitive 'backspace' actions to clear a text field instead of a bulk delete) and navigation inefficiencies (e.g., manually clicking through directories when a search bar is present).",
    "Parameter Vector Miscalibration": "The agent correctly identifies the required interaction type (e.g., scroll, hscroll, swipe, left_click_drag) but critically fails to parameterize its physical vector. This manifests as either magnitude insufficiency (outputting the correct mathematical sign but a drastically undersized scale like 5 `pixels` instead of 200, resulting in negligible UI movement) or a polarity/direction reversal (the agent reasons or executes the exact opposite direction of the goal, e.g., writing 'Action: scroll down' but mistakenly outputting a positive pixels value that physically scrolls up, or swapping the coordinate arrays in a swipe)",
    "Visual Hallucination": "The agent attempts to interact with a UI element, icon, or menu that strictly does not exist in the current visual observation. The agent fails to ground its action in the current screen pixels and instead relies on false internal priors. This manifests either as a temporal ghost interaction, where the agent attempts to interact with an element that existed in a previous state (trajectory history) but has since disappeared or been closed (e.g., clicking the 'X' of a modal that is no longer on screen), or as a prior-bias hallucination, where the agent blindly interacts with a coordinate based on standard UI layouts (e.g., assuming a 'Search' bar or 'Settings' gear is in the top right corner) without visually confirming it actually exists in the current UI rendering.",
    "Timing and Latency Neglect": "The agent executes a logically correct subsequent action, but does so prematurely by failing to account for asynchronous UI changes. It completely ignores visual indicators that the system is busy (e.g., loading spinners, progress bars, disabled 'processing' buttons) or mid-transition (e.g., unfolding menus, sliding panels).",
    "Constraint Neglect": "The user instruction explicitly dictates multifaceted requirements. The agent executes an action that completely ignores the explicit text constraints by selecting a visually available, highly distracting alternative. For example, if the user instruction says 'click the 2012 report', the agent clicks the '2023' report visible on the screen. The rejected target MUST be a plausible UI element that currently exists in the visual observation.",
    "Action-Operation Misalignment": "The agent's verbalized intent drastically contradicts its executable JSON string. It reasons correctly and writes a highly logical 'Action:' description for the user instruction, but the actual `<tool_call>` it generates is completely different and nonsensical for that intent.",
}


consequence_instructions = {
    "Action Formulation Error": "Predict an immediate parser crash with ZERO structural UI change.",
    "Grounding/Spatial Error": "Predict the action hitting 'dead space' (resulting in ZERO UI change) OR accidentally triggering the nearest adjacent, incorrect UI element because the visual marker is placed in empty space or on the wrong element.",
    "Procedural Prerequisite Neglect": "Predict a silent failure (e.g., typing into an unfocused field), interacting with a blocking foreground overlay instead of the target, or acting prematurely on an unresolved loading state.",
    "Semantic Error": "Predict the successful execution of the *wrong* action, leading the UI down an incorrect, diverging navigational path.",
    "Termination Misjudgment": "Predict the agent either halting abruptly while the UI remains incomplete or answer to the task is not done (Premature), OR executing a redundant/incorrect UI action that alters an already successfully completed state (Over-Execution).",
    "Constraint Neglect": "Predict the UI updating to reflect a partially correct state that explicitly violates the requested attributes or positions (e.g., opening a post by the wrong author, or inserting text at the wrong location).",
    "Observation Neglect": "Predict the UI unnecessarily shifting (e.g., scrolling away, opening menus) and obscuring the target information that was already perfectly visible.",
    "Suboptimal Path": "Predict the UI undergoing a painfully inefficient micro-change (e.g., a single arrow click or single backspace) instead of the expected global UI shift (e.g., direct keyboard entry or Ctrl+A).",
    "Parameter Vector Miscalibration": "Predict the UI moving in the exact opposite direction (Polarity Reversal) or moving an imperceptible, microscopic amount resulting in no meaningful layout change (Magnitude Insufficiency).",
    "Visual Hallucination": "Predict a silent failure or a 'dead click', as the visual marker is placed on a strictly non-existent coordinate, empty space, or a ghost element.",
    "Timing and Latency Neglect": "Predict the input being completely ignored or dropped by the system because the UI is locked in a loading state, processing spinner, or mid-transition animation.",
    "Action-Operation Misalignment": "Predict a completely unexpected and nonsensical UI transition because the executed tool_call strictly contradicts the logical intent."
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
        action.group(1).strip() if action else None,
        structural.group(1).strip() if structural else None
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


def warmup_rationale(task_sample):
    task = task_sample.get("task", "")
    trajectory = task_sample.get("trajectory", [])
    proposed_step = task_sample.get("rejected_proposed_step", "")
    error_dimension = task_sample.get("dimension_violated", "None")
    image_path = task_sample.get("image_path", "")
    base64_image = encode_image_to_base64(image_path)

    chosen_operation = task_sample.get("operation", "")
    chosen_action = task_sample.get("action", "")
    transition_rationale = task_sample.get("transition_rationale", "")
    action_mechanism, structural_effect = extract_parts(transition_rationale)

    if task_sample['env'] == 'android':
        platform = 'mobile'
    elif task_sample['env'] in ['web', 'ubuntu', 'windows', 'mac']:
        platform = 'desktop'
    else:
        ValueError(f"{task_sample['env']} is not defined.")

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

    if error_dimension not in error_dimensions:
        return None

    system_prompt = rejected_action_rationale_system_prompt.format(platform_action_space=action_spaces[platform], error_dimension_str=f"**{error_dimension}**: {error_dimensions[error_dimension]}", target_consequence_instruction=consequence_instructions[error_dimension])

    user_prompt = f"User Instruction (Goal):\n{task}\n\n"
    user_prompt += f"Trajectory:\n{trajectory_str}\n\n"
    user_prompt += f"Current Observation:\n<image>\n\n"
    user_prompt += f"Current Proposed Step:\n{proposed_step}\n\n"
    user_prompt += f"**Target Error Dimension**:\n{error_dimension}\n\n"
    user_prompt += f"**Reference Proposed Step**:\nAction: {chosen_operation}\n\n"
    user_prompt += f"**Reference Action Mechanism**:\n{action_mechanism}\n\n"
    user_prompt += f"**Reference Structural Effect**:\n{structural_effect}\n\n"

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

    # local_save_dir = args.local_dir
    # os.makedirs(local_save_dir, exist_ok=True)
    # output_path = os.path.join(local_save_dir, "critic_rejected_dataset_fixed_grounding_vector_scale.jsonl")

    # annotation_dir = args.annotation_dir
    # pattern = os.path.join(annotation_dir, "double_blind_filtered_negative_samples*.jsonl")
    # files = glob.glob(pattern)

    # print(files)  # sanity check

    # df = pd.concat(
    #     [pd.read_json(f, lines=True) for f in files],
    #     ignore_index=True
    # )

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    output_path = os.path.join(local_save_dir, "critic_rejected_dataset_fixed_grounding_merged_all_new_no_verbal.jsonl")

    annotation_dir = args.annotation_dir
    df = pd.read_json(os.path.join(annotation_dir, "double_blind_filtered_negative_samples.jsonl"), lines=True)

    df['state_transition'] = df['state_transition'].fillna("N/A")
    df['transition_rationale'] = df['transition_rationale'].fillna("N/A")

    # Remove invalid errors
    errors_to_remove = [
        "Memory & State Tracking Error",
        "Action-Operation Misalignment",
        "Grounding/Spatial Error"
    ]
    df = df[~df["dimension_violated"].isin(errors_to_remove)]

    # Controll the termination misjudgment item
    df_term = df[df["dimension_violated"] == "Termination Misjudgment"]
    df_other = df[df["dimension_violated"] != "Termination Misjudgment"]

    df_term_downsampled = df_term.sample(n=1600, random_state=42)
    df = pd.concat([df_other, df_term_downsampled]).sample(frac=1).reset_index(drop=True)

    # Add more variation on the vector error
    df_scroll = df[df["rejected_proposed_step"].astype(str).str.contains(
            r'"action"\s*:\s*"scroll"',
            case=False,
            na=False,
            regex=True
        )
    ]
    df_other = df[~df["rejected_proposed_step"].astype(str).str.contains(
            r'"action"\s*:\s*"scroll"',
            case=False,
            na=False,
            regex=True
        )
    ]
    df_scroll["rejected_proposed_step"] = df_scroll["rejected_proposed_step"].apply(
        lambda x: add_mouse_move_before_scroll(str(x), prob=0.3)
    )
    df = pd.concat([df_other, df_scroll]).sample(frac=1).reset_index(drop=True)

    # Filter invalid format just in case
    df_format = df[df["dimension_violated"] == "Action Formulation Error"]
    df_other = df[df["dimension_violated"] != "Action Formulation Error"]

    mask_valid = df["rejected_proposed_step"].apply(are_all_tool_calls_valid)
    df_other = df_other[mask_valid]
    df = pd.concat([df_format, df_other]).sample(frac=1).reset_index(drop=True)


    # ==========================================
    # NEW: Safe Selection for No-Intent Ablation
    # ==========================================
    # Define the dimensions that STRICTLY REQUIRE verbal intent to be diagnosed
    unsafe_dimensions = [
        "Grounding/Spatial Error",
        "Visual Hallucination",
        "Action-Operation Misalignment"
    ]

    # Create a boolean mask identifying rows that are SAFE to have their intent removed
    is_safe_condition = ~df["dimension_violated"].astype(str).str.contains('|'.join(unsafe_dimensions), case=False, na=False)
    
    safe_df = df[is_safe_condition]

    # Calculate exactly how many rows make up 20% of the total dataset
    num_no_intent_samples = int(len(df) * 0.30)

    # Ensure we actually have enough safe samples to cover the 20%
    if len(safe_df) < num_no_intent_samples:
        print(f"WARNING: Not enough safe samples ({len(safe_df)}) to meet the 20% quota ({num_no_intent_samples}). Using all safe samples.")
        num_no_intent_samples = len(safe_df)

    # Sample the 20% strictly from the safe subset
    df_no_intent = safe_df.sample(n=num_no_intent_samples, random_state=42)
    df = df_no_intent
    df['rejected_proposed_step'] = df['rejected_proposed_step'].apply(extract_tool_call_only)

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
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

