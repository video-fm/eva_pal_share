#!/usr/bin/env python3
"""Test script for trajectory prediction using trajectory_v0 or trajectory_v1."""

import os
from dataclasses import dataclass
from typing import Literal

import tyro


@dataclass
class Args:
    """Trajectory prediction script arguments."""

    traj_version: Literal[0, 1] = 1
    """Trajectory model version (0 or 1)."""
    model: str = "gpt-4o-mini"
    """OpenAI model for v1 (step extraction and trajectory)."""
    gemini_model: str = "gemini-robotics-er-1.5-preview"
    """Gemini model for object detection and trajectory."""


def main(args: Args):

    caption_to_image = {
        "Move the orange juice to the left.": "move_orange_juice_to_left.jpg",
        "Place pineapple in bowl": "place_pineapple_in_bowl.jpg",
        "Place watermelom in bowl": "place_water_melon_in_bowl.jpg",
        "Stack cups in green, red, yellow order": "stack_cups_in_green_red_yellow.jpg",
    }
    base_dir = "/home/franka/eva_jiani/data/test_traj"
    img_dir = os.path.join(base_dir, "test_traj_examples")
    suffix = "_v1" if args.traj_version == 1 else "_v0"
    resized_dir = os.path.join(base_dir, f"resized_examples{suffix}")
    traj_dir = os.path.join(base_dir, f"traj_examples{suffix}")
    os.makedirs(resized_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    if args.traj_version == 0:
        _run_v0(args, caption_to_image, img_dir, resized_dir, traj_dir)
    else:
        _run_v1(args, caption_to_image, img_dir, resized_dir, traj_dir)

    print("Done")


def _run_v0(args, caption_to_image, img_dir, resized_dir, traj_dir):
    from eva.detectors.trajectory_v0 import get_image_resized, query_target_trajectory

    for caption, img_filename in caption_to_image.items():
        img_path = os.path.join(img_dir, img_filename)
        resized_path = os.path.join(resized_dir, img_filename.replace(".jpg", "_resized.jpg"))
        img = get_image_resized(img_path)
        img.save(resized_path)

        save_path = os.path.join(traj_dir, img_filename.replace(".jpg", "_traj.jpg"))
        query_target_trajectory(
            img, caption, model_name=args.gemini_model, save_path=save_path, visualize=True
        )


def _run_v1(args, caption_to_image, img_dir, resized_dir, traj_dir):
    from openai import OpenAI

    from eva.detectors.traj_vis_utils import add_arrow
    from eva.detectors.trajectory_v1 import (
        encode_pil_image,
        get_image_resized,
        query_target_location,
        query_target_objects,
        query_trajectory,
    )

    client = OpenAI()

    for caption, img_filename in caption_to_image.items():
        img_path = os.path.join(img_dir, img_filename)
        resized_path = os.path.join(resized_dir, img_filename.replace(".jpg", "_resized.jpg"))
        img = get_image_resized(img_path)
        img.save(resized_path)

        base_save_path = os.path.join(traj_dir, img_filename.replace(".jpg", "_traj"))
        img_encoded = encode_pil_image(img)

        target = query_target_objects(client, caption, model=args.model, img_encoded=img_encoded)
        for step_idx, step in enumerate(target["steps"]):
            manipulating_object = step["manipulating_object"]
            target_related_object = step["target_related_object"]
            target_location = step["target_location"]
            target_objects = list(set([manipulating_object, target_related_object]))

            object_locations = query_target_location(
                img, target_objects, model_name=args.gemini_model, visualize=False
            )
            if object_locations is None:
                continue

            manipulating_object_point = object_locations.get(manipulating_object)
            target_related_object_point = object_locations.get(target_related_object)
            if manipulating_object_point is None or target_related_object_point is None:
                continue

            trajectory = query_trajectory(
                client,
                img=img,
                img_encoded=img_encoded,
                task=step["step"],
                manipulating_object=manipulating_object,
                manipulating_object_point=manipulating_object_point,
                target_related_object=target_related_object,
                target_related_object_point=target_related_object_point,
                target_location=target_location,
                model_name=args.model,
            )

            if trajectory and "trajectory" in trajectory and len(trajectory["trajectory"]) >= 2:
                pts = [tuple(p) for p in trajectory["trajectory"]]
                img_with_arrow = add_arrow(img, pts, color="red", line_width=3)
                save_path = f"{base_save_path}_{step_idx}.jpg" if step_idx > 0 else f"{base_save_path}.jpg"
                img_with_arrow.convert("RGB").save(save_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
