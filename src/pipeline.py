import logging
from src.utils.control_result_paths import new_path_results, metadata_save
from src.dataloader.div_data import get_dataloader
from src.yolo.funcs_yolov11 import train_model

def run_experiment(model_used, data_used, BASE_DIR, data_create, train_yolo=False):


    # Divide los sets
    get_dataloader(model_used, data_used, BASE_DIR, data_create)

    if train_yolo:
        logging.info("Comienza el entrenamiento")
        train_model(model_used, BASE_DIR)
        logging.info("Se realizó correctamente el entrenamiento")
        logging.info("Se guardó correctamente el modelo")
