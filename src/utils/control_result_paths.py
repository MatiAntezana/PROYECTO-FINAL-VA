import os
import re


def new_path_results(type_model_user, data_user, features_user):
    """Create a new numbered results folder and write selected run parameters."""
    target_dir = os.path.join("..", "Model-Automatization-CarSales", "results")
    pattern = re.compile(r"model_(\d+)_folder")
    existing_numbers = []

    for folder_name in os.listdir(target_dir):
        match = pattern.match(folder_name)
        if match:
            existing_numbers.append(int(match.group(1)))

    new_number = max(existing_numbers, default=0) + 1
    new_folder_name = f"model_{new_number}_folder"
    new_folder_path = os.path.join(target_dir, new_folder_name)

    file_content = "Files used for model\n"
    file_content += f"Type model: {type_model_user}\n"
    file_content += f"Data user: {data_user}\n"
    file_content += f"Features user: {features_user}\n"

    os.makedirs(new_folder_path)
    txt_file_path = os.path.join(new_folder_path, f"param_used_{new_number}.txt")
    with open(txt_file_path, "w") as file_txt:
        file_txt.write(file_content)

    return new_folder_path


def metadata_save(metadata):
    """Placeholder to persist metadata for a completed run."""
    return "a"
