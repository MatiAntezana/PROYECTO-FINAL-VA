import os
import re

def new_path_results(type_model_user, data_user, features_user):
    target_dir = os.path.join("..", "Model-Automatization-CarSales", "results")

    pattern = re.compile(r"model_(\d+)_folder")
    existing_numbers = []
    
    for folder_name in os.listdir(target_dir):
        match = pattern.match(folder_name)
        if match:
            existing_numbers.append(int(match.group(1)))
    
    new_number = max(existing_numbers, default=0) + 1
    new_folder_name = f"model_{new_number}_folder"
    
    new_folder_route = os.path.join(target_dir, new_folder_name)
    
    contain_file_info = "Files used for model\n"

    contain_file_info += f"Type model: {type_model_user}\n"
    contain_file_info += f"Data user: {data_user}\n"
    contain_file_info += f"Features user: {features_user}\n"
    
    os.makedirs(new_folder_route)
    file_txt_folder = os.path.join(new_folder_route, f"param_used_{new_number}.txt")
    
    with open(file_txt_folder, 'w') as file_txt:
        file_txt.write(contain_file_info)

    return new_folder_route

def metadata_save(metadata):
    return "a"