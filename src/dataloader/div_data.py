import io
import logging
import os
import random
import shutil
from shutil import copy2

import yaml
from PIL import Image


def delete_file_folder(folder_path):
    """Delete all files and subfolders inside a folder."""
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    print(f"All content in '{folder_path}' was removed.")


def rename_img_txt(source_folder, base_name):
    """Load paired RGB/thermal images and return normalized label references."""
    file_names = os.listdir(source_folder)
    renamed_images = []
    renamed_txt_files = []
    index = 1

    rgb_images = [name for name in file_names if name.lower().endswith(".jpg") and "_R" not in name]
    thermal_images = [name for name in file_names if name.lower().endswith(".jpg") and "_R" in name]

    for rgb_image_name in sorted(rgb_images):
        try:
            rgb_image_path = os.path.join(source_folder, rgb_image_name)
            with open(rgb_image_path, "rb") as image_file:
                rgb_image_data = Image.open(io.BytesIO(image_file.read())).copy()

            rgb_txt_path = os.path.splitext(rgb_image_path)[0] + ".txt"
            rgb_txt_file = rgb_txt_path if os.path.exists(rgb_txt_path) else None
            new_rgb_txt_name = f"{base_name}_{str(index).zfill(4)}.txt"

            image_number = os.path.splitext(rgb_image_name)[0].split("_")[-1]
            thermal_number = str(int(image_number) - 1).zfill(4)
            thermal_image_name = next((name for name in thermal_images if thermal_number in name), None)

            thermal_image_data = None
            thermal_txt_file = None
            new_thermal_txt_name = None

            if thermal_image_name:
                thermal_image_path = os.path.join(source_folder, thermal_image_name)
                with open(thermal_image_path, "rb") as image_file:
                    thermal_image_data = Image.open(io.BytesIO(image_file.read())).copy()

                thermal_txt_path = os.path.splitext(thermal_image_path)[0] + ".txt"
                thermal_txt_file = thermal_txt_path if os.path.exists(thermal_txt_path) else None
                new_thermal_txt_name = f"{base_name}_{str(index).zfill(4)}_T.txt"

            renamed_images.append((rgb_image_data, thermal_image_data))
            renamed_txt_files.append(
                ((new_rgb_txt_name, rgb_txt_file), (new_thermal_txt_name, thermal_txt_file))
            )
            index += 1
        except Exception as error:
            print(f"Error processing {rgb_image_name}: {error}")

    return renamed_images, renamed_txt_files


def construct_file_yaml(output_path, train_path, valid_path, test_path, classes, base_dir, yaml_name):
    """Create a YOLO data YAML file under the src directory."""
    data = {
        "path": f"{base_dir}/{output_path}",
        "train": f"{base_dir}/{output_path}{train_path}",
        "val": f"{base_dir}/{output_path}{valid_path}",
        "test": f"{base_dir}/{output_path}{test_path}",
        "names": classes,
    }
    file_path = f"src/{yaml_name}"
    with open(file_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, allow_unicode=True)


def split_dataset_images(images, txt_files, output_path, seed=341, train_ratio=0.7, valid_ratio=0.2):
    """Split paired RGB/thermal samples into train, valid, and test subsets."""
    for split_name in ["train", "valid", "test"]:
        for image_type in ["img_normal", "thermal"]:
            os.makedirs(os.path.join(output_path, split_name, image_type, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_path, split_name, image_type, "labels"), exist_ok=True)

    random.seed(seed)
    dataset_items = list(zip(images, txt_files))
    random.shuffle(dataset_items)

    total_items = len(dataset_items)
    train_end = int(total_items * train_ratio)
    valid_end = train_end + int(total_items * valid_ratio)

    split_sets = {
        "train": dataset_items[:train_end],
        "valid": dataset_items[train_end:valid_end],
        "test": dataset_items[valid_end:],
    }

    for split_name, split_data in split_sets.items():
        for index, ((rgb_image, thermal_image), (rgb_txt, thermal_txt)) in enumerate(split_data):
            base_name = f"{split_name}_{index + 1}"
            rgb_image_name = f"{base_name}.jpg"
            thermal_image_name = f"{base_name}_T.jpg"
            rgb_txt_name = f"{base_name}.txt"
            thermal_txt_name = f"{base_name}_T.txt"

            rgb_image_output = os.path.join(output_path, split_name, "img_normal", "images", rgb_image_name)
            thermal_image_output = os.path.join(output_path, split_name, "thermal", "images", thermal_image_name)
            rgb_txt_output = os.path.join(output_path, split_name, "img_normal", "labels", rgb_txt_name)
            thermal_txt_output = os.path.join(output_path, split_name, "thermal", "labels", thermal_txt_name)

            rgb_image.save(rgb_image_output, format="JPEG")
            thermal_image.save(thermal_image_output, format="JPEG")
            copy2(rgb_txt[1], rgb_txt_output)
            copy2(thermal_txt[1], thermal_txt_output)

    print("Dataset split completed.")


def split_dataset_one_images(images, txt_files, output_path, model_type, seed=341, train_ratio=0.7, valid_ratio=0.2):
    """Split samples for single-stream models (RGB or thermal only)."""
    for split_name in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_path, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, split_name, "labels"), exist_ok=True)

    random.seed(seed)
    dataset_items = list(zip(images, txt_files))
    random.shuffle(dataset_items)

    total_items = len(dataset_items)
    train_end = int(total_items * train_ratio)
    valid_end = train_end + int(total_items * valid_ratio)

    split_sets = {
        "train": dataset_items[:train_end],
        "valid": dataset_items[train_end:valid_end],
        "test": dataset_items[valid_end:],
    }

    for split_name, split_data in split_sets.items():
        for index, ((rgb_image, thermal_image), (rgb_txt, thermal_txt)) in enumerate(split_data):
            base_name = f"{split_name}_{index + 1}"
            rgb_image_name = f"{base_name}.jpg"
            thermal_image_name = f"{base_name}_T.jpg"
            rgb_txt_name = f"{base_name}.txt"
            thermal_txt_name = f"{base_name}_T.txt"

            rgb_image_output = os.path.join(output_path, split_name, "images", rgb_image_name)
            thermal_image_output = os.path.join(output_path, split_name, "images", thermal_image_name)
            rgb_txt_output = os.path.join(output_path, split_name, "labels", rgb_txt_name)
            thermal_txt_output = os.path.join(output_path, split_name, "labels", thermal_txt_name)

            rgb_image.save(rgb_image_output, format="JPEG")
            copy2(rgb_txt[1], rgb_txt_output)

            if model_type == "BASE":
                thermal_image.save(thermal_image_output, format="JPEG")
                copy2(thermal_txt[1], thermal_txt_output)

    print("Dataset split completed.")


def delete_file_yaml(folder_path):
    """Remove all YAML files from a folder."""
    file_names = os.listdir(folder_path)
    yaml_files = [file_name for file_name in file_names if file_name.lower().endswith(".yaml")]

    for yaml_file in yaml_files:
        file_path = os.path.join(folder_path, yaml_file)
        os.remove(file_path)
        print(f"Removed file: {file_path}")

    logging.info("YAML files were removed.")


def dataloader_diff(model_used, data_used, base_dir):
    """Create train/valid/test splits and generate YAML files per model type."""
    param_sets = data_used.param_sets

    delete_file_folder(data_used.output_path)
    cow_images, cow_txt_files = rename_img_txt(data_used.dataset_cow, "Cow")
    deer_images, deer_txt_files = rename_img_txt(data_used.dataset_deer, "Deer")
    horse_images, horse_txt_files = rename_img_txt(data_used.dataset_horse, "Horse")
    images = cow_images + deer_images + horse_images
    txt_files = cow_txt_files + deer_txt_files + horse_txt_files
    delete_file_yaml("src")

    if model_used.name_model == "yolov11_late_fusion":
        split_dataset_images(
            images,
            txt_files,
            data_used.output_path,
            seed=param_sets["seed"],
            train_ratio=param_sets["train"],
            valid_ratio=param_sets["valid"],
        )
        construct_file_yaml(
            data_used.output_path,
            "/train/img_normal/images",
            "/valid/img_normal/images",
            "/test/img_normal/images",
            data_used.classes,
            base_dir,
            "IMGS_l_f.yaml",
        )
        construct_file_yaml(
            data_used.output_path,
            "/train/thermal/images",
            "/valid/thermal/images",
            "/test/thermal/images",
            data_used.classes,
            base_dir,
            "TERM_l_f.yaml",
        )
    elif model_used.name_model == "yolov11_base":
        split_dataset_one_images(
            images,
            txt_files,
            data_used.output_path,
            model_type="BASE",
            seed=param_sets["seed"],
            train_ratio=param_sets["train"],
            valid_ratio=param_sets["valid"],
        )
        construct_file_yaml(
            data_used.output_path,
            "/train",
            "/valid",
            "/test",
            data_used.classes,
            base_dir,
            "BASE_yolov11.yaml",
        )
    elif model_used.name_model == "yolov11_HST":
        split_dataset_one_images(
            images,
            txt_files,
            data_used.output_path,
            model_type="HST",
            seed=param_sets["seed"],
            train_ratio=param_sets["train"],
            valid_ratio=param_sets["valid"],
        )
        construct_file_yaml(
            data_used.output_path,
            "/train",
            "/valid",
            "/test",
            data_used.classes,
            base_dir,
            "IMG_HST.yaml",
        )
    elif model_used.name_model == "yolov11_GST":
        split_dataset_one_images(
            images,
            txt_files,
            data_used.output_path,
            model_type="GST",
            seed=param_sets["seed"],
            train_ratio=param_sets["train"],
            valid_ratio=param_sets["valid"],
        )
        construct_file_yaml(
            data_used.output_path,
            "/train",
            "/valid",
            "/test",
            data_used.classes,
            base_dir,
            "IMG_GST.yaml",
        )


def dataloder_diff(model_used, data_used, BASE_DIR):
    """Backward-compatible alias for dataloader_diff."""
    dataloader_diff(model_used, data_used, BASE_DIR)


def get_dataloader(model_used, data_used, base_dir, data_create):
    """Prepare dataset splits when requested by the pipeline."""
    if data_create:
        dataloader_diff(model_used, data_used, base_dir)
        logging.info("Datasets were split successfully.")
    else:
        logging.info("Dataset split step was skipped.")
