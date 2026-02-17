import os
import re
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from ultralytics import YOLO


def replace_images_with_labels(folder_path):
    """Return the label folder path for a given image folder path."""
    folder_path = str(folder_path)
    return folder_path.replace("images", "labels")


def extract_txt_files_from_folder(folder_path):
    """Read and sort TXT files from a folder by sample index."""
    txt_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            txt_files.append(os.path.join(folder_path, file_name))

    return sorted(txt_files, key=lambda path: int(os.path.basename(path).split("_")[1].split(".")[0]))


def extract_first_column(file_path):
    """Extract label IDs from the first column of a YOLO label TXT file."""
    labels = []
    with open(file_path, "r") as file:
        for line in file:
            columns = line.split()
            labels.append(int(columns[0]))
    return labels


def iou(box1, box2):
    """Compute IoU between two bounding boxes in xyxy format."""
    x_min_intersection = max(box1[0], box2[0])
    y_min_intersection = max(box1[1], box2[1])
    x_max_intersection = min(box1[2], box2[2])
    y_max_intersection = min(box1[3], box2[3])

    intersection_area = max(0, x_max_intersection - x_min_intersection) * max(
        0, y_max_intersection - y_min_intersection
    )
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def IOU(box1, box2):
    """Backward-compatible alias for iou."""
    return iou(box1, box2)


def center_calc(box):
    """Compute the center point of a bounding box."""
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return np.array([x_center, y_center])


def match_boxes(boxes_rgb, boxes_thermal, iou_threshold, max_distance):
    """Match RGB and thermal boxes using IoU and center-distance constraints."""
    rgb_count, thermal_count = len(boxes_rgb), len(boxes_thermal)
    scores = np.full((rgb_count, thermal_count), -np.inf, dtype=float)

    for rgb_idx, rgb_box in enumerate(boxes_rgb):
        for thermal_idx, thermal_box in enumerate(boxes_thermal):
            overlap = iou(rgb_box, thermal_box)
            distance = np.linalg.norm(center_calc(rgb_box) - center_calc(thermal_box))

            if overlap >= iou_threshold and distance <= max_distance:
                scores[rgb_idx, thermal_idx] = overlap

    matches = []
    for rgb_idx in range(rgb_count):
        best_thermal_idx = None
        if thermal_count > 0:
            candidate_idx = int(np.argmax(scores[rgb_idx]))
            if np.isfinite(scores[rgb_idx, candidate_idx]):
                best_thermal_idx = candidate_idx
        matches.append((rgb_idx, best_thermal_idx))

    return matches


def get_jpg_files_from_folder(folder_path):
    """Load JPG files from a folder sorted by numeric suffix."""
    path = Path(folder_path)
    jpg_files = list(path.glob("*.jpg"))

    def extract_number(file_path):
        file_name = file_path.stem
        return int(file_name.split("_")[1])

    sorted_files = sorted(jpg_files, key=extract_number)
    return [cv2.imread(str(image_path)) for image_path in sorted_files]


def plot_confusion_matrix(y_true, y_pred, labels, path_save, include_extra_label=False):
    """Plot and save the confusion matrix as a PDF."""
    confusion = confusion_matrix(y_true, y_pred, labels=labels).T
    class_names = ["Cow", "Deer", "Horse"]
    if include_extra_label:
        class_names.append("Extra labels")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion,
        annot=True,
        cmap="coolwarm",
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 16},
    )
    plt.ylabel("Predicted")
    plt.xlabel("True")
    plt.savefig(path_save, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()


def obtain_images_txt(path_yaml_1, path_yaml_2):
    """Load paired test images and corresponding TXT label files from YAML definitions."""
    with open(path_yaml_1, "r") as file:
        data_1 = yaml.safe_load(file)
    with open(path_yaml_2, "r") as file:
        data_2 = yaml.safe_load(file)

    rgb_images = get_jpg_files_from_folder(data_1["test"])
    thermal_images = get_jpg_files_from_folder(data_2["test"])
    labels_path = replace_images_with_labels(data_1["test"])
    txt_files = extract_txt_files_from_folder(labels_path)
    return rgb_images, thermal_images, txt_files


def list_boxes_class(pred_rgb, pred_thermal):
    """Extract boxes and class-confidence tuples from YOLO predictions."""
    rgb_boxes = []
    thermal_boxes = []
    rgb_classes = []
    thermal_classes = []

    for rgb_result in pred_rgb:
        for rgb_box in rgb_result.boxes:
            x1_rgb, y1_rgb, x2_rgb, y2_rgb = map(int, rgb_box.xyxy[0])
            rgb_boxes.append([x1_rgb, y1_rgb, x2_rgb, y2_rgb])
            class_id = int(rgb_box.cls)
            rgb_classes.append((class_id, float(rgb_box.conf[0])))

    for thermal_result in pred_thermal:
        for thermal_box in thermal_result.boxes:
            x1_thermal, y1_thermal, x2_thermal, y2_thermal = map(int, thermal_box.xyxy[0])
            thermal_boxes.append([x1_thermal, y1_thermal, x2_thermal, y2_thermal])
            class_id = int(thermal_box.cls)
            thermal_classes.append((class_id, float(thermal_box.conf[0])))

    return rgb_boxes, thermal_boxes, rgb_classes, thermal_classes


def get_union_box(box1, box2):
    """Create a union box that contains two input boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    union_x1 = min(x1_1, x1_2)
    union_y1 = min(y1_1, y1_2)
    union_x2 = max(x2_1, x2_2)
    union_y2 = max(y2_1, y2_2)
    return [union_x1, union_y1, union_x2, union_y2]


def obtain_predictions(model_normal, model_thermal, img_normal, img_thermal, min_iou, max_distance, conf_threshold):
    """Generate late-fusion predictions from RGB and thermal detections."""
    pred_rgb = model_normal.predict(img_normal)
    pred_thermal = model_thermal.predict(img_thermal)

    rgb_boxes, thermal_boxes, rgb_classes, thermal_classes = list_boxes_class(pred_rgb, pred_thermal)
    matched_indices = match_boxes(rgb_boxes, thermal_boxes, min_iou, max_distance)

    raw_matches = {}
    for match_idx, (rgb_box_idx, thermal_box_idx) in enumerate(matched_indices):
        if thermal_box_idx is not None:
            if rgb_classes[rgb_box_idx][0] == thermal_classes[thermal_box_idx][0]:
                raw_matches[match_idx] = [
                    rgb_boxes[rgb_box_idx],
                    thermal_boxes[thermal_box_idx],
                    rgb_classes[rgb_box_idx][1],
                ]
            else:
                raw_matches[match_idx] = None
        elif rgb_classes[rgb_box_idx][1] >= conf_threshold:
            raw_matches[match_idx] = [
                rgb_boxes[rgb_box_idx],
                rgb_boxes[rgb_box_idx],
                rgb_classes[rgb_box_idx][1],
            ]
        else:
            raw_matches[match_idx] = None

    final_predictions = {}
    merged_boxes_conf = {}
    for index, element in raw_matches.items():
        if element is not None:
            box1, box2, confidence = element
            final_predictions[index] = rgb_classes[index][0]
            merged_boxes_conf[index] = [get_union_box(box1, box2), confidence]
        else:
            final_predictions[index] = None
            merged_boxes_conf[index] = None

    return final_predictions, merged_boxes_conf


def fill_with_opposite_elements(smaller_list, larger_list):
    """Pad predictions using fallback opposite classes until lengths match."""
    opposite_element = {0: 1, 2: 1, 1: 2}

    while len(smaller_list) < len(larger_list):
        for element in larger_list:
            if len(smaller_list) >= len(larger_list):
                break
            if element in opposite_element:
                smaller_list.append(opposite_element[element])

    return smaller_list


def fuse_predictions(result_predictions, true_labels):
    """Align model predictions with ground-truth labels for evaluation."""
    predicted_labels = [value for value in result_predictions.values() if value is not None]
    prediction_count = len(predicted_labels)
    true_label_count = len(true_labels)

    if prediction_count < true_label_count:
        completed_predictions = fill_with_opposite_elements(predicted_labels, true_labels)
        return completed_predictions, 0
    if prediction_count == true_label_count:
        return predicted_labels, 0

    return predicted_labels[:true_label_count], predicted_labels[true_label_count:]


def labels_predictions(model_normal, model_thermal, path_yaml_1, path_yaml_2, min_iou, max_distance, conf_threshold):
    """Run late-fusion predictions for all test samples and return evaluation vectors."""
    rgb_images, thermal_images, txt_files = obtain_images_txt(path_yaml_1, path_yaml_2)
    y_true = []
    y_pred = []
    extra_labels = []

    for rgb_image, thermal_image, txt_file in zip(rgb_images, thermal_images, txt_files):
        true_labels = extract_first_column(txt_file)
        predictions, _ = obtain_predictions(
            model_normal,
            model_thermal,
            rgb_image,
            thermal_image,
            min_iou,
            max_distance,
            conf_threshold,
        )
        final_predictions, sample_extra_labels = fuse_predictions(predictions, true_labels)
        y_true += true_labels
        y_pred += final_predictions
        if sample_extra_labels != 0:
            extra_labels += sample_extra_labels

    return y_true, y_pred, extra_labels


def aggregate_labels(y_true, y_pred, extra_labels):
    """Append synthetic ground-truth class for unmatched extra predictions."""
    for _ in range(len(extra_labels)):
        y_true.append(3)
    y_pred += extra_labels
    return y_true, y_pred


def agregate_labels(y_true, y_pred, more_labels):
    """Backward-compatible alias for aggregate_labels."""
    return aggregate_labels(y_true, y_pred, more_labels)


def evaluate_model_late(
    model_normal,
    model_thermal,
    path_yaml_1,
    path_yaml_2,
    min_iou,
    max_distance=None,
    conf_threshold=None,
    path_save=None,
    **legacy_kwargs,
):
    """Evaluate late-fusion predictions and save a confusion matrix plot."""
    if max_distance is None:
        max_distance = legacy_kwargs.pop("distancia_max", None)
    if conf_threshold is None:
        conf_threshold = legacy_kwargs.pop("thread_conf", None)
    if path_save is None:
        path_save = legacy_kwargs.pop("path_save", None)
    if max_distance is None or conf_threshold is None or path_save is None:
        raise ValueError("max_distance, conf_threshold, and path_save are required.")

    labels = [0, 1, 2]
    y_true, y_pred, extra_labels = labels_predictions(
        model_normal,
        model_thermal,
        path_yaml_1,
        path_yaml_2,
        min_iou,
        max_distance,
        conf_threshold,
    )

    precision_per_class = precision_score(y_true, y_pred, average=None, labels=range(len(labels)))
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=range(len(labels)))
    accuracy = accuracy_score(y_true, y_pred)

    print("Global accuracy:", accuracy)
    print("Global precision:", sum(precision_per_class) / 3)
    print("Global recall:", sum(recall_per_class) / 3)

    if len(extra_labels) > 0:
        labels = [0, 1, 2, 3]
        y_true, y_pred = aggregate_labels(y_true, y_pred, extra_labels)

    plot_confusion_matrix(y_true, y_pred, labels, path_save, include_extra_label=bool(extra_labels))


def find_latest_tunex_folder(start_dir):
    """Find the latest tune folder produced by Ultralytics tuning runs."""
    latest_number = -1
    latest_folder = None

    for root, dirs, _ in os.walk(start_dir):
        for dir_name in dirs:
            if dir_name == "tune":
                latest_folder = os.path.join(root, dir_name)
            elif dir_name.startswith("tune"):
                match = re.match(r"tune(\d+)", dir_name)
                if match:
                    number = int(match.group(1))
                    if number > latest_number:
                        latest_number = number
                        latest_folder = os.path.join(root, dir_name)

    return latest_folder


def copy_best_hyperparameters(latest_folder, dest_dir):
    """Copy best hyperparameter YAML from tuning output to destination path."""
    file_path = os.path.join(latest_folder, "best_hyperparameters.yaml")
    if os.path.exists(file_path):
        shutil.copy(file_path, dest_dir)
        print(f"File copied to {dest_dir}")
    else:
        print("File 'best_hyperparameters.yaml' was not found.")


def read_yaml(path_yaml):
    """Read and parse a YAML file."""
    with open(path_yaml, "r") as file:
        return yaml.safe_load(file)


def delete_all_folders(path):
    """Delete all subfolders inside the provided path."""
    if os.path.exists(path) and os.path.isdir(path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")
    else:
        print(f"Path '{path}' is not a valid directory.")


def create_model(model_used, base_dir, path_yaml, new_model_name):
    """Tune and train a YOLO model using best discovered hyperparameters."""
    model = YOLO("src/yolo11s.pt")
    model.tune(
        data=path_yaml,
        epochs=model_used.epochs_hiper,
        iterations=model_used.iterations_hiper,
        optimizer=model_used.optimizer_hiper,
        plots=False,
        save=True,
        val=True,
        project=f"{base_dir}/models/hiper",
    )

    latest_folder = find_latest_tunex_folder(f"{base_dir}/models/hiper")
    copy_best_hyperparameters(latest_folder, "models/best_params.yaml")
    delete_all_folders(f"{base_dir}/models/hiper")

    best_params = read_yaml("models/best_params.yaml")

    model.train(
        data=path_yaml,
        epochs=model_used.epochs_train,
        batch=model_used.batch,
        imgsz=model_used.imgsz,
        device=model_used.device,
        name=new_model_name,
        project=model_used.project,
        save_period=model_used.save_period,
        lr0=best_params["lr0"],
        lrf=best_params["lrf"],
        momentum=best_params["momentum"],
        weight_decay=best_params["weight_decay"],
        warmup_epochs=best_params["warmup_epochs"],
        warmup_momentum=best_params["warmup_momentum"],
        box=best_params["box"],
        cls=best_params["cls"],
        dfl=best_params["dfl"],
        hsv_h=best_params["hsv_h"],
        hsv_s=best_params["hsv_s"],
        hsv_v=best_params["hsv_v"],
        degrees=best_params["degrees"],
        translate=best_params["translate"],
        scale=best_params["scale"],
        shear=best_params["shear"],
        perspective=best_params["perspective"],
        flipud=best_params["flipud"],
        fliplr=best_params["fliplr"],
        bgr=best_params["bgr"],
        mosaic=best_params["mosaic"],
        mixup=best_params["mixup"],
        copy_paste=best_params["copy_paste"],
    )


def train_model(model_used, base_dir):
    """Train the configured model variant."""
    if model_used.name_model == "yolov11_late_fusion":
        create_model(model_used, base_dir, f"{base_dir}/src/IMGS_l_f.yaml", model_used.name_model_normal)
        create_model(model_used, base_dir, f"{base_dir}/src/TERM_l_f.yaml", model_used.name_model_thermal)
    elif model_used.name_model == "yolov11_base":
        create_model(model_used, base_dir, f"{base_dir}/src/BASE_yolov11.yaml", model_used.name)
    elif model_used.name_model == "yolov11_GST":
        create_model(model_used, base_dir, f"{base_dir}/src/IMG_GST.yaml", model_used.name)
    elif model_used.name_model == "yolov11_HST":
        create_model(model_used, base_dir, f"{base_dir}/src/IMG_HST.yaml", model_used.name)
