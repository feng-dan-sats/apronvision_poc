import os
import argparse
import boto3
import json
import shutil
import random
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, f1_score

def download_s3_file(bucket, key, local_path):
    s3 = boto3.client('s3')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)

def download_s3_files(bucket, keys, prefix, local_dir):
    s3 = boto3.client('s3')
    for key in keys:
        rel_path = os.path.relpath(key, prefix)
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)

def parse_manifest_and_get_keys(manifest_local_path):
    image_keys = []
    with open(manifest_local_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            s3_uri = data['source-ref']
            # Get S3 key from URI (assumes format s3://bucket/key)
            key = '/'.join(s3_uri.split('/')[3:])
            image_keys.append(key)
    return image_keys

def convert_manifest_to_yolo(manifest_path, image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    with open(manifest_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            image_name = os.path.basename(data['source-ref'])
            annotations = data.get('annotations', {}).get('bounding-box', {}).get('boxes', [])
            label_file = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))
            with open(label_file, 'w') as label_out:
                for box in annotations:
                    class_id = box.get("class_id", 0)
                    x_center = (box['left'] + box['width'] / 2) / data['image-size'][0]['width']
                    y_center = (box['top'] + box['height'] / 2) / data['image-size'][0]['height']
                    w = box['width'] / data['image-size'][0]['width']
                    h = box['height'] / data['image-size'][0]['height']
                    label_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def evaluate(model, image_dir, label_dir, class_names):
    y_true, y_pred = [], []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for fname in image_files:
        image_path = os.path.join(image_dir, fname)
        label_path = os.path.join(label_dir, fname.rsplit('.', 1)[0] + '.txt')
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            labels = [int(line.split()[0]) for line in f.readlines()]
        if not labels:
            continue
        results = model(image_path, verbose=False)
        preds = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
        if not preds:
            continue
        n = min(len(labels), len(preds))
        y_true.extend(labels[:n])
        y_pred.extend(preds[:n])

    metrics = {}
    if y_true and y_pred:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        total = cm.sum(axis=1)
        correct = np.diag(cm)
        acc = correct / (total + 1e-6)
        f1 = f1_score(y_true, y_pred, average=None, labels=list(range(len(class_names))))
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        overall_acc = np.mean(np.array(y_true) == np.array(y_pred))

        print(f"confusion_matrix={json.dumps(cm.tolist())};")
        for i, name in enumerate(class_names):
            print(f"{name}_accuracy={acc[i]:.4f};")
            print(f"{name}_f1={f1[i]:.4f};")
        print(f"F1_score={macro_f1:.4f};")
        print(f"Accuracy_total={overall_acc:.4f};")

        metrics["confusion_matrix"] = cm.tolist()
        metrics["per_class"] = {
            name: {"accuracy": float(acc[i]), "f1_score": float(f1[i])}
            for i, name in enumerate(class_names)
        }
        metrics["overall"] = {
            "f1_macro": float(macro_f1),
            "accuracy_total": float(overall_acc)
        }

        public_output_dir = Path("/opt/ml/output/public")
        public_output_dir.mkdir(parents=True, exist_ok=True)
        evaluation_path = public_output_dir / "evaluation.json"
        with open(evaluation_path, "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        print("Not enough predictions to evaluate.")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--image-prefix', type=str, required=True)
    parser.add_argument('--manifest-prefix', type=str, required=True)
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()

    base_dir = Path("/opt/ml/input/data/train")
    raw_images = base_dir / "raw_images"
    raw_labels = base_dir / "raw_labels"
    output_base = base_dir / "yolo_dataset"
    manifest_path = base_dir / "output.manifest"

    # 1. Download manifest file
    manifest_s3_key = f"{args.manifest_prefix}/output.manifest".lstrip('/')
    download_s3_file(args.bucket, manifest_s3_key, str(manifest_path))

    # 2. Parse manifest to get image keys
    image_keys = parse_manifest_and_get_keys(str(manifest_path))

    # 3. Download only the referenced images
    download_s3_files(args.bucket, image_keys, args.image_prefix, str(raw_images))

    print(f"✅ Downloaded {len(image_keys)} images from manifest")

    # 4. Convert manifest to YOLO label format
    convert_manifest_to_yolo(manifest_path, raw_images, raw_labels)

    image_files = sorted([f for f in os.listdir(raw_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    random.shuffle(image_files)
    n_total = len(image_files)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.2)
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    print(f"📊 Dataset split: Total={n_total}, Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    paths = {
        "train/images": output_base / "images/train",
        "val/images": output_base / "images/val",
        "test/images": output_base / "images/test",
        "train/labels": output_base / "labels/train",
        "val/labels": output_base / "labels/val",
        "test/labels": output_base / "labels/test"
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    def copy_files(file_list, split):
        for img in file_list:
            name = os.path.splitext(img)[0]
            shutil.copy(raw_images / img, paths[f"{split}/images"] / img)
            label_path = raw_labels / f"{name}.txt"
            if label_path.exists():
                shutil.copy(label_path, paths[f"{split}/labels"] / f"{name}.txt")

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    data_yaml = output_base / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write(f"""path: {output_base}
train: images/train
val: images/val
names:
  0: empty_pd_sats
  1: empty_pd_others
  2: loaded_pd
  3: empty_ct_sats
  4: empty_ct_others
  5: loaded_ct
""")

    model = YOLO(args.model)
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz)

    best_model_path = Path("runs/detect/train/weights/best.pt")
    model_output_path = Path("/opt/ml/model/best.pt")
    public_output_dir = Path("/opt/ml/output/public")
    public_output_dir.mkdir(parents=True, exist_ok=True)
    if best_model_path.exists():
        shutil.copy(best_model_path, model_output_path)
        shutil.copy(best_model_path, public_output_dir / "best.pt")
    if Path("runs/detect/train").exists():
        shutil.make_archive(str(public_output_dir / "yolo_logs"), 'zip', "runs/detect/train")

    class_names = [
        "empty_pd_sats", "empty_pd_others", "loaded_pd",
        "empty_ct_sats", "empty_ct_others", "loaded_ct"
    ]
    metrics = evaluate(YOLO(str(best_model_path)), str(paths["val/images"]), str(paths["val/labels"]), class_names)
    print(metrics)

if __name__ == "__main__":
    main()



