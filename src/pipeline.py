import logging
from src.dataloader.div_data import get_dataloader
from src.yolo.funcs_yolov11 import train_model


def run_experiment(model_used, data_used, BASE_DIR, data_create, train_yolo=False):
    """Run dataset preparation and optional YOLO training."""
    get_dataloader(model_used, data_used, BASE_DIR, data_create)

    if train_yolo:
        logging.info("Training started.")
        train_model(model_used, BASE_DIR)
        logging.info("Training completed successfully.")
        logging.info("Model artifacts were saved.")
