# inference_processing.py

import os
import sys
import subprocess
def install_requirements(requirements_path="/opt/ml/processing/input/requirements.txt"):
    if os.path.exists(requirements_path):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])

install_requirements()

import tarfile
import json
import argparse
from pathlib import Path
import boto3
from ultralytics import YOLO
from PIL import Image


def extract_model(model_tar_path, extract_dir):
    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print(f"Extracted model to {extract_dir}")

def download_images_from_s3(bucket, prefix, local_dir):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    os.makedirs(local_dir, exist_ok=True)

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith((".jpg", ".jpeg", ".png")):
                local_path = os.path.join(local_dir, os.path.basename(key))
                s3.download_file(bucket, key, local_path)
                print(f"Downloaded {key}")

def run_inference(model_path, image_dir, output_json_path):
    model = YOLO(model_path)
    results = {}

    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, image_file)
            prediction = model(image_path)[0]

            bbox_counts = { 
                'empty_pd_sats': 0,
                'empty_pd_others': 0,
                'loaded_pd': 0,
                'empty_ct_sats': 0,
                'empty_ct_others': 0,
                'loaded_ct': 0
            }

            for box in prediction.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                bbox_counts[class_name] += 1

            results[image_file] = bbox_counts
            print(f"Processed {image_file}")
            print(bbox_counts)

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_json_path}")

def upload_to_s3(file_path, bucket, key):
    s3 = boto3.client("s3")
    s3.upload_file(file_path, bucket, key)
    print(f"Uploaded {file_path} to s3://{bucket}/{key}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tar_path", type=str)
    parser.add_argument("--image_bucket", type=str)
    parser.add_argument("--image_prefix", type=str)
    parser.add_argument("--output_bucket", type=str)
    parser.add_argument("--output_key", type=str)
    args = parser.parse_args()

    model_dir = "/opt/ml/processing/model"
    image_dir = "/opt/ml/processing/images"
    output_json_path = "/opt/ml/processing/output/results.json"

    extract_model(args.model_tar_path, model_dir)
    model_path = os.path.join(model_dir, "best.pt")  # adjust if different filename

    download_images_from_s3(args.image_bucket, args.image_prefix, image_dir)
    run_inference(model_path, image_dir, output_json_path)
    upload_to_s3(output_json_path, args.output_bucket, args.output_key)

if __name__ == "__main__":
    main()
