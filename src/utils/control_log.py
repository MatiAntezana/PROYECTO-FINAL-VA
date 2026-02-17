import os


def define_n_log():
    """Return the next sequential log file name in the logs directory."""
    log_directory = os.path.join("..", "PROYECTO_FINAL_VA", "logs")
    file_names = os.listdir(log_directory)
    formatted_files = [
        file_name
        for file_name in file_names
        if file_name.startswith("loger_") and file_name.endswith(".log")
    ]

    if not formatted_files:
        return "loger_1.log"

    file_numbers = [int(file_name.split("_")[1].split(".")[0]) for file_name in formatted_files]
    max_number = max(file_numbers)
    return f"loger_{max_number + 1}.log"
