name_model = "yolov11_late_fusion"

epochs_train = 100
epochs_hiper = 40
iterations_hiper = 300
optimizer_hiper = "AdamW"

batch = 8
imgsz = 640
device = "gpu"
name_model_normal = "yolov11_L_F_NORMAL"
name_model_thermal = "yolov11_L_F_THERMAL"
project = "models/yolov11_L_F"
save_period = epochs_train
