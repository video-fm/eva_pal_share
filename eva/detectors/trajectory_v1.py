import argparse
import os
import json
import re
from openai import OpenAI

from google import genai
from google.genai import types

import textwrap
import sys
from pathlib import Path

from PIL import Image
import io
import numpy as np
import base64

from eva.detectors.traj_vis_utils import add_arrow, get_image_resized



GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
google_client = genai.Client(api_key=GOOGLE_API_KEY)

prompt_target_template = """
You are an assistant that, given a natural language instruction for a robot, identifies:
1) What is the plan for the robot to complete the task?
2) For each step in the plan, identify the object that the robot must manipulate and the target location for that object.

Rules:
- "manipulating_object" must be only the physical item the robot grasps or moves, not the target location or container.
- "target_location" must describe where that object should be placed, applied, or moved (including relations like "to the left of the banana" or "onto the blue plate").
- If multiple objects are mentioned, choose the single primary object that is directly manipulated.
- If no clear target location is given, use the empty string "" for "target_location".

Example 1
Caption: Add sprinkles to the Coke in the designated container.
Output:
{{
  "steps": [
    {{
      "step": "Add sprinkles to the Coke in the designated container.",
      "manipulating_object": "sprinkles",
      "target_location": "Coke in the designated container",
      "target_related_object": "Coke"
    }}
  ]
}}

Example 2
Caption: Place the pineapple to the left of the banana, ensuring they are positioned neatly.
Output:
{{
  "steps": [
    {{
      "step": "Place the pineapple to the left of the banana, ensuring they are positioned neatly.",
      "manipulating_object": "pineapple",
      "target_location": "left of the banana",
      "target_related_object": "banana"
    }}
  ]
}}

Example 3
Caption: Move the pineapple to the left.
Output:
{{
  "steps": [
    {{
      "step": "Move the pineapple to the left.",
      "manipulating_object": "pineapple",
      "target_location": "left of the pineapple",
      "target_related_object": "pineapple"
    }}
  ]
}}

Now process this caption:
Caption: {caption}

Return ONLY a single valid JSON object (no extra text). Use this format:

{{
  "steps": [
    {{
      "step": "...",
      "manipulating_object": "...",
      "target_location": "..."
      "target_related_object": "..."
    }}, 
    {{
      "step": "...",
      "manipulating_object": "...",
      "target_location": "..."
      "target_related_object": "..."
    }},
    ...
  ]
}}
"""

trajectory_generation_prompt_template = """
You are a spatial reasoning and motion planning assistant for tabletop manipulation. 

   INPUT: 
   - Task instruction: string
   - Image: an image of the tabletop with the objects in the scene.
   - Image height and width: int, int
   - target object to manipulate location: (x, y) 
   - relevant object location: (x, y) # an object's location that is related to the target position.
   
   GOAL: Predict a safe 2D movement for the target object on the tabletop and output a trajectory arrow. 

   REQUIREMENTS: 
   1. Interpret spatial meaning of the instruction (e.g., right, left, toward object, away from edge, etc.). 
   2. Note that the movement should be relative to the aspect of the robot, not the camera.
   3. Detect potential risks such as: - Falling off table / outside workspace - Collision with obstacles 
   4. Choose a **safe reachable target point** that satisfies the instruction while staying on the table.
   5. Prefer keeping object inside central tabletop region when possible. 
   6. The trajectory shall be a multi-step trajectory, with at least 2 points.
   
    TRAJECTORY RULES:
   - Use at least 3 points
   - First point: The target object location to manipulate.
   - Final point: settle at the predicted target location.
  
   OUTPUT: 
   - reasoning: brief explanation of safety + direction 
   - start point: (x, y) # input start point
   - target location: (x, y) # safe predicted goal 
   - trajectory: a list of points [(x, y), (x, y), ...] # safe predicted trajectory from start point to target location

  DO NOT:
  - Use camera/image left-right
  - Simply subtract a constant from x without reasoning

  Now process this task:
  Task: {task}
  Image height: {height}
  Image width: {width}
  The object to manipulate is {manipulating_object} at: {manipulating_object_point}
  The related object to the target position is {target_related_object}, which located at: {target_related_object_point}
  We want to move the manipulating object to the target location: {target_location}
  
  Return ONLY a single valid JSON object (no extra text). Use this format:
  {{
    "reasoning": "...",
    "start_point": [x, y],
    "end_point": [x, y],
    "trajectory": [[x, y], [x, y], ...]
  }}
"""



step_completion_prompt_template = """
You are a spatial reasoning and motion planning assistant for tabletop manipulation. 

We are currently performing this step in the task: {task}.

Based on the previous images and the current image, please determine if the step is complete.

Return ONLY a single valid JSON object (no extra text). Use this format:
{{
  "reasoning": "...",
  "is_complete": true/false
}}
"""

trajectory_generation_prompt_template_with_target_location = """
You are a spatial reasoning and motion planning assistant for tabletop manipulation. 

   INPUT: 
   - Task instruction: string
   - Image: an image of the tabletop with the objects in the scene.
   - Image height and width: int, int
   - target object to manipulate location: (x, y) 
   - end point: (x, y) # the given target location.
   
   GOAL: Predict a safe 2D movement for the target object on the tabletop and output a trajectory arrow. 

   REQUIREMENTS: 
   1. Interpret spatial meaning of the instruction (e.g., right, left, toward object, away from edge, etc.). 
   2. Note that the movement should be relative to the aspect of the robot, not the camera.
   3. Detect potential risks such as: - Falling off table / outside workspace - Collision with obstacles 
   4. Choose a **safe reachable target point** that satisfies the instruction while staying on the table.
   5. Prefer keeping object inside central tabletop region when possible. 
   6. The trajectory shall be a multi-step trajectory, with at least 2 points.
   
    TRAJECTORY RULES:
   - Use at least 3 points
   - First point: The target object location to manipulate.
   - Final point: settle at the target location.
  
   OUTPUT: 
   - reasoning: brief explanation of safety + direction 
   - start point: (x, y) # input start point
   - target location: (x, y) # input goal location
   - trajectory: a list of points [(x, y), (x, y), ...] # safe predicted trajectory from start point to target location

  DO NOT:
  - Use camera/image left-right
  - Simply subtract a constant from x without reasoning

  Now process this task:
  Task: {task}
  Image height: {height}
  Image width: {width}
  The object to manipulate is {manipulating_object} at: {manipulating_object_point}
  The related object to the target position is {target_related_object}, which located at: {target_related_object_point}
  We want to move the manipulating object to the target location: {target_location}, which is at: {target_location_point}
  
  Return ONLY a single valid JSON object (no extra text). Use this format:
  {{
    "reasoning": "...",
    "start_point": [x, y],
    "end_point": [x, y],
    "trajectory": [[x, y], [x, y], ...]
  }}
"""

def encode_image(img_path: str) -> str:
    """Convert image file to base64 string."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_pil_image(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_json(json_output):
  # Parsing out the markdown fencing
  lines = json_output.splitlines()
  for i, line in enumerate(lines):
    if line == "```json":
      # Remove everything before "```json"
      json_output = "\n".join(lines[i + 1 :])
      # Remove everything after the closing "```"
      json_output = json_output.split("```")[0]
      break  # Exit the loop once "```json" is found
  return json_output
     

def call_gemini_robotics_er(img, prompt, config=None, model_name= "gemini-robotics-er-1.5-preview"):
    default_config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=100)
    )

    if config is None:
        config = default_config

    image_response = google_client.models.generate_content(
          model=model_name,
          contents=[img, prompt],
          config=config,
    )

    print(image_response.text)
    return parse_json(image_response.text)



def query_target_objects(client: OpenAI, caption: str, model: str = "gpt-4o-mini") -> list[str]:
    """Query ChatGPT to extract target objects from a caption."""
    prompt = prompt_target_template.format(caption=caption)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
    )
    text = response.choices[0].message.content.strip()
    text = parse_json(text)
    return json.loads(text)


def query_target_location(img, queries: list[str], model_name: str = "gemini-robotics-er-1.5-preview", visualize: bool = False) -> dict[str, tuple[float, float]]:
    
    point_prompt = textwrap.dedent(f"""\
    Get all points matching the following objects: {', '.join(queries)}. The label
    returned should be an identifying name for the object detected.
    
    Note that there shall be multiple table corners in the image.

    The answer should follow the JSON format:
    [{{"point": [y_norm, x_norm], "label": "object-name"}}, ...]

    The points are in [y, x] format normalized to 0-1000.
    """)

    json_output = call_gemini_robotics_er(img, point_prompt, model_name=model_name)
    width, height = img.size
    
    points_data = []
    try:
        data = json.loads(json_output)
        points_data.extend(data)
    except json.JSONDecodeError:
        print("Warning: Invalid JSON response. Skipping.")
        return None

    object_locations = {}
    for obj_location_info in data:
        obj_name = obj_location_info["label"]
        (yNorm, xNorm) = obj_location_info["point"]  # [y, x] format per gemini_vis_util
        x = (xNorm / 1000.0) * width
        y = (yNorm / 1000.0) * height
        object_locations[obj_name] = (x, y)
    
    return object_locations

def query_step_completion(
  client: OpenAI, 
  img_list, 
  img_encoded_list, 
  step: str, 
  model_name: str = "gpt-4o-mini",
  max_images: int = 3) -> dict:
  
    if len(img_list) > max_images:
        indices = np.linspace(0, len(img_list) - 1, max_images, dtype=int)
        img_list = [img_list[i] for i in indices]
        img_encoded_list = [img_encoded_list[i] for i in indices]

    prompt = step_completion_prompt_template.format(task=step)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user",
                   "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_encoded}"
                            },
                        }
                        for img_encoded in img_encoded_list
                    ],
                  ]}],
        
        temperature=0.0,
        max_tokens=200,
    )
    
    text = response.choices[0].message.content.strip()
    text = parse_json(text)
    json_output = json.loads(text)
    return json_output

def query_trajectory(
    client: OpenAI,
    img,
    img_encoded: str,
    task: str,
    manipulating_object: str,
    manipulating_object_point: str,
    target_related_object: str,
    target_related_object_point: str,
    target_location: str,
    model_name: str = "gpt-4o-mini",
    target_location_point: str = None,
) -> str:

    if target_location_point is None:
        prompt = trajectory_generation_prompt_template.format(
            task=task,
            height=img.height,
            width=img.width,
            manipulating_object=manipulating_object,
            manipulating_object_point=manipulating_object_point,
            target_related_object=target_related_object,
            target_related_object_point=target_related_object_point,
            target_location=target_location,
        )
    else: 
        prompt = trajectory_generation_prompt_template_with_target_location.format(
          task=task,
          height=img.height,
          width=img.width,
          manipulating_object=manipulating_object,
          manipulating_object_point=manipulating_object_point,
          target_related_object=target_related_object,
          target_related_object_point=target_related_object_point,
          target_location=target_location,
          target_location_point=target_location_point,
      )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_encoded}"
                        },
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=200,
    )

    text = response.choices[0].message.content.strip()
    text = parse_json(text)
    json_output = json.loads(text)
    return json_output


def get_sorted_frame_paths(img_dir: str) -> list[str]:
    """Return numerically sorted paths for raw frames (e.g. 000.jpg, 001.jpg, …)."""
    filenames = [f for f in os.listdir(img_dir) if re.match(r"^\d{3}\.jpg$", f)]
    filenames.sort()
    return [os.path.join(img_dir, f) for f in filenames]


def parse_args():
    
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--gemini-model", default="gemini-robotics-er-1.5-preview")
    p.add_argument("--test-pipeline-dir", default="/home/jianih/research/STSG-ICL/data/ICL_VLA_small/test_traj_prediction")
    p.add_argument("--img-path", default=None)
    p.add_argument("--save-resized-img-path", default=None)
    p.add_argument("--save-trajectory-img-path", default="/home/jianih/research/STSG-ICL/data/ICL_VLA_small/stsg_visualizations")
    p.add_argument("--check-interval", type=int, default=20,
                   help="Check step completion every N frames")
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    caption_dir_lookup = {
        # "Move the orange juice to the left.": "2025_08_29_move_orange_juice_to_left_human/Fri_Aug_29_10_24_28_2025",
        # "Place pineapple in bowl": "2025_08_28_place_pineapple_in_bowl_human/Thu_Aug_28_16_06_09_2025",
        # "Place watermelom in bowl": "2025_08_28_place_watermelon_in_bowl_human/Thu_Aug_28_16_14_14_2025",
        "Stack cups in green, red, yellow order": "2025_10_07_stack_the_cups_in_green_red_yellow_order/Tue_Oct_07_17_29_58_2025",
    }
    
    default_img_dir_template = "/home/jianih/research/STSG-ICL/data/ICL_VLA_small/test/L2_out_of_domain/{}/recordings/frames/varied_camera_1/"

    client = OpenAI()

    results = []
    for caption, img_dir in caption_dir_lookup.items():
      
        default_img_dir = default_img_dir_template.format(img_dir)
        frame_paths = get_sorted_frame_paths(default_img_dir)

        target = query_target_objects(client, caption, model=args.model)
        steps = target["steps"]

        step_idx = 0
        current_step = steps[step_idx]
        current_trajectory = None
        current_end_point = None
        img_history = []
        img_encoded_history = []

        for frame_idx, frame_path in enumerate(frame_paths):
            img = get_image_resized(frame_path)
            img_encoded = encode_pil_image(img)
            img_history.append(img)
            img_encoded_history.append(img_encoded)

            if current_trajectory is None:
                manipulating_object = current_step["manipulating_object"]
                target_related_object = current_step["target_related_object"]
                target_objects = list(set([manipulating_object, target_related_object]))

                object_locations = query_target_location(
                    img, target_objects, model_name=args.gemini_model, visualize=False
                )

                manipulating_object_point = object_locations[manipulating_object]
                target_related_object_point = object_locations[target_related_object]

                current_trajectory = query_trajectory(
                    client,
                    img=img,
                    img_encoded=img_encoded,
                    task=current_step["step"],
                    manipulating_object=manipulating_object,
                    manipulating_object_point=manipulating_object_point,
                    target_related_object=target_related_object,
                    target_related_object_point=target_related_object_point,
                    target_location=current_step["target_location"],
                    model_name=args.model,
                )
                current_end_point = current_trajectory["end_point"]

                if args.save_trajectory_img_path is not None:
                    pts = [tuple(p) for p in current_trajectory["trajectory"]]
                    if len(pts) >= 2:
                        traj_save_path = os.path.join(
                            default_img_dir,
                            f"{os.path.splitext(os.path.basename(frame_path))[0]}_trajectory.jpg",
                        )
                        img_with_arrow = add_arrow(img, pts, color="red", line_width=3)
                        img_with_arrow.convert("RGB").save(traj_save_path)

            if len(img_history) >= args.check_interval and len(img_history) % args.check_interval == 0:
                completion = query_step_completion(
                    client, img_history, img_encoded_history,
                    step=current_step["step"], model_name=args.model,
                )

                print(f"[frame {frame_idx}] step {step_idx} "
                      f"(\"{current_step['step']}\"): is_complete={completion['is_complete']}")

                if completion["is_complete"]:
                    results.append({
                        "caption": caption,
                        "step": current_step,
                        "trajectory": current_trajectory,
                        "completed_at_frame": frame_idx,
                    })

                    step_idx += 1
                    if step_idx >= len(steps):
                        break

                    current_step = steps[step_idx]
                    current_trajectory = None
                    current_end_point = None

                    img_history = [img]
                    img_encoded_history = [img_encoded]

                else:
                    manipulating_object = current_step["manipulating_object"]
                    target_related_object = current_step["target_related_object"]
                    target_objects = list(set([manipulating_object, target_related_object]))

                    object_locations = query_target_location(
                        img, target_objects, model_name=args.gemini_model, visualize=False
                    )

                    manipulating_object_point = object_locations[manipulating_object]
                    target_related_object_point = object_locations[target_related_object]

                    current_trajectory = query_trajectory(
                        client,
                        img=img,
                        img_encoded=img_encoded,
                        task=current_step["step"],
                        manipulating_object=manipulating_object,
                        manipulating_object_point=manipulating_object_point,
                        target_related_object=target_related_object,
                        target_related_object_point=target_related_object_point,
                        target_location=current_step["target_location"],
                        model_name=args.model,
                        target_location_point=current_end_point,
                    )

                    if args.save_trajectory_img_path is not None:
                        pts = [tuple(p) for p in current_trajectory["trajectory"]]
                        if len(pts) >= 2:
                            traj_save_path = os.path.join(
                                default_img_dir,
                                f"{os.path.splitext(os.path.basename(frame_path))[0]}_trajectory.jpg",
                            )
                            img_with_arrow = add_arrow(img, pts, color="red", line_width=3)
                            img_with_arrow.convert("RGB").save(traj_save_path)

        if current_trajectory is not None:
            results.append({
                "caption": caption,
                "step": current_step,
                "trajectory": current_trajectory,
                "completed_at_frame": None,
            })

    print(json.dumps(results, indent=2))
    print("Done")