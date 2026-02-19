import argparse
import os
import json
from openai import OpenAI

from google import genai
from google.genai import types

import base64
from PIL import Image

from eva.detectors.traj_vis_utils import add_arrow, get_image_resized

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
google_client = genai.Client(api_key=GOOGLE_API_KEY)

def encode_image(img_path: str) -> str:
    """Convert image file to base64 string."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")



def save_trajectory_img(img: Image.Image, save_path: str) -> None:
    """Save image with trajectory overlay."""
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(save_path)


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


def query_target_trajectory(img, task: str, model_name: str = "gemini-robotics-er-1.5-preview", save_path: str = "points.jpg", visualize: bool = False) -> list[str]:

    prompt = f"""
        Task: {task}
        Generate 10 points for the trajectory.
        The points should be labeled by order of the trajectory, from '0'
        from the target object to <n> (final desired point)
        The answer should follow the json format:
        [{{"point": <point>, "label": <label1>}}, ...].
        The points are in [y, x] format normalized to 0-1000.
        """

    json_output = call_gemini_robotics_er(img, prompt, model_name=model_name)
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

    if visualize and len(object_locations) >= 2:
        ordered_points = [
            object_locations[str(i)]
            for i in sorted(int(k) for k in object_locations.keys())
        ]
        img_with_arrow = add_arrow(img, ordered_points, color="red", line_width=3)
        save_trajectory_img(img_with_arrow, save_path)
    elif visualize:
        save_trajectory_img(img, save_path)

    return object_locations


def parse_args():
    base_dir = "/home/franka/eva_jiani/data/test_traj"
    default_img_dir = os.path.join(base_dir, "test_traj_examples")
    default_resized_dir = os.path.join(base_dir, "resized_examples")
    default_traj_dir = os.path.join(base_dir, "traj_examples")

    default_img_path = os.path.join(default_img_dir, "move_orange_juice_to_left.jpg")
    default_resized_img_path = os.path.join(default_resized_dir, "000_resized.jpg")
    default_trajectory_img_path = os.path.join(default_traj_dir, "000_trajectory.jpg")

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--gemini-model", default="gemini-robotics-er-1.5-preview")
    p.add_argument("--test-pipeline-dir", default=default_traj_dir)
    p.add_argument("--save-location-img-file-name", default="points.jpg")
    p.add_argument("--img-path", default=default_img_path)
    p.add_argument("--save-resized-img-path", default=default_resized_img_path)
    p.add_argument("--save-trajectory-img-path", default=default_trajectory_img_path)
    p.add_argument("--traj-version", type=int, choices=[0, 1], default=0, help="Trajectory model version (0 or 1)")
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    caption_to_image = {
        "Move the orange juice to the left.": "move_orange_juice_to_left.jpg",
        "Place pineapple in bowl": "place_pineapple_in_bowl.jpg",
        "Place watermelom in bowl": "place_water_melon_in_bowl.jpg",
        "Stack cups in green, red, yellow order": "stack_cups_in_green_red_yellow.jpg",
    }
    base_dir = "/home/franka/eva_jiani/data/test_traj"
    img_dir = os.path.join(base_dir, "test_traj_examples")
    resized_dir = os.path.join(base_dir, "resized_examples")
    traj_dir = os.path.join(base_dir, "traj_examples")
    os.makedirs(resized_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    client = OpenAI()

    results = []
    for caption, img_filename in caption_to_image.items():
        img_path = os.path.join(img_dir, img_filename)
        resized_path = os.path.join(resized_dir, img_filename.replace(".jpg", "_resized.jpg"))
        img = get_image_resized(img_path)
        img.save(resized_path)

        save_path = os.path.join(traj_dir, img_filename.replace(".jpg", "_points.jpg"))
        trajectory = query_target_trajectory(img, caption, model_name=args.gemini_model, save_path=save_path, visualize=True)
        
    print("Done")

