from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import cv2
import numpy as np
import shutil
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch import Tensor, IntTensor
from torchmetrics.detection import MeanAveragePrecision

def replace_images_with_labels(folder_path):
    # Convertir la ruta a cadena (si no lo es ya)
    folder_path = str(folder_path)
    
    # Reemplaza "images" por "labels" en la ruta de la carpeta
    new_folder_path = folder_path.replace('images', 'labels')

    return new_folder_path

def extract_txt_files_from_folder(folder_path):
    txt_files = []
    # Recorre todos los archivos en la carpeta
    for filename in os.listdir(folder_path):
        # Verifica si el archivo tiene la extensión .txt
        if filename.endswith('.txt'):
            txt_files.append(os.path.join(folder_path, filename))
    
    # Ordena los archivos por el número después de 'test_'
    txt_files_sorted = sorted(
        txt_files,
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    
    return txt_files_sorted

def extract_first_column(file_path):
    first_column = []
    # Abre el archivo en modo de lectura
    with open(file_path, 'r') as file:
        for line in file:
            # Divide cada línea en columnas
            columns = line.split()
            # Agrega el primer valor (de la primera columna) a la lista
            first_column.append(int(columns[0]))
    
    return first_column

def IOU(box1, box2):
    # Coordenadas de la intersección
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # Áreas
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def center_calc(box):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return np.array([x_center, y_center])

def match_boxes(boxes_normal, boxes_thermal, iou_threshold, distancia_max):
    n, t = len(boxes_normal), len(boxes_thermal)
    costos = np.ones((n, t)) * np.inf

    control = 0

    for i, box_n in enumerate(boxes_normal):
        for j, box_t in enumerate(boxes_thermal):
            iou = IOU(box_n, box_t)
            distancia = np.linalg.norm(center_calc(box_n) - center_calc(box_t))

            if iou >= iou_threshold and distancia <= distancia_max:
                costos[i, j] = iou
                control += 1
            else:
                costos[i, j] = None

    matches_final = []

    for idx_match in range(n):
        best_iou = 0
        idx_best = None
        for j in range(t):
            if costos[idx_match, j] > best_iou:
                best_iou = costos[idx_match, j]
                idx_best = j

        matches_final.append((idx_match, idx_best))


    return matches_final


def get_jpg_files_from_folder(folder_path):
    path = Path(folder_path)
    
    jpg_files = list(path.glob("*.jpg"))

    def extract_number(file_path):
    # Suponiendo que el formato del archivo es test_x.jpg, donde x es un número
        file_name = file_path.stem  # Obtener el nombre del archivo sin la extensión
        number = int(file_name.split('_')[1])
        return number

    # Ordenar los archivos por el número extraído
    sorted_files = sorted(jpg_files, key=extract_number)
    # print(sorted_files)
    images = [cv2.imread(str(img)) for img in sorted_files]
    return images

def plot_confusion_matrix(y_true, y_pred, labels, path_save, more_column=False):

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.T

    labels = ["Cow", "Deer", "Horse"]

    if more_column:
        labels = labels + ["Extra labels"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="coolwarm", fmt='d', xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 16})  # Cambia el tamaño de los números
    plt.ylabel('Predicted')
    plt.xlabel('True')

    plt.savefig(path_save, format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()  #

def obtain_images_txt(path_yaml_1, path_yaml_2):
    with open(path_yaml_1, 'r') as file:
        data_1 = yaml.safe_load(file)

    with open(path_yaml_2, 'r') as file:
        data_2 = yaml.safe_load(file)

    images_normal = get_jpg_files_from_folder(data_1["test"])
    images_termal = get_jpg_files_from_folder(data_2["test"])

    path_txt = replace_images_with_labels(data_1["test"])

    list_txt = extract_txt_files_from_folder(path_txt)

    return images_normal, images_termal, list_txt

def list_boxes_class(pred_normal, pred_termal):
    list_boxes_norm = []
    list_boxes_term = []

    list_class_norm = []
    list_class_term = []

    for res_norm in pred_normal:
        for box_normal in res_norm.boxes:
            x1_nor, y1_nor, x2_nor, y2_nor = map(int, box_normal.xyxy[0])
            list_boxes_norm.append([x1_nor, y1_nor, x2_nor, y2_nor])
            class_id = int(box_normal.cls)
            list_class_norm.append((class_id, float(box_normal.conf[0])))

    for res_term in pred_termal:
        for box_term in res_term.boxes:
            x1_ter, y1_ter, x2_ter, y2_ter = map(int, box_term.xyxy[0])
            list_boxes_term.append([x1_ter, y1_ter, x2_ter, y2_ter])
            class_id = int(box_term.cls)
            list_class_term.append((class_id, float(box_term.conf[0])))

    return list_boxes_norm, list_boxes_term, list_class_norm, list_class_term

def get_union_box(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xu1 = min(x1_1, x1_2)
    yu1 = min(y1_1, y1_2)
    xu2 = max(x2_1, x2_2)
    yu2 = max(y2_1, y2_2)

    return [xu1, yu1, xu2, yu2]

def obtain_predictions(model_normal, model_termal, img_normal, img_termal, min_iou, distancia_max, thread_conf):

    pred_normal = model_normal.predict(img_normal)
    pred_termal = model_termal.predict(img_termal)

    list_boxes_norm, list_boxes_term, list_class_norm, list_class_term = list_boxes_class(pred_normal, pred_termal)

    result_matcheo = match_boxes(list_boxes_norm, list_boxes_term, min_iou, distancia_max)

    dic_boxes = {}

    idx_dic = 0

    for idx_normal_box, idx_termal_box in result_matcheo:
        if idx_termal_box != None:
            # print("Es:",list_class_norm, list_class_term)
            if list_class_norm[idx_normal_box][0] == list_class_term[idx_termal_box][0]:
                dic_boxes[idx_dic] = [list_boxes_norm[idx_normal_box], list_boxes_term[idx_termal_box], list_class_norm[idx_normal_box][1]]
            else:
                # NO PASA
                dic_boxes[idx_dic] = None
        else:
            if list_class_norm[idx_normal_box][1] >= thread_conf:
                dic_boxes[idx_dic] = [list_boxes_norm[idx_normal_box], list_boxes_norm[idx_normal_box], list_class_norm[idx_normal_box][1]]
                idx_dic += 1
            else:
                # NO PASA
                dic_boxes[idx_dic] = None
        idx_dic += 1
    
    dic_result_final = {}


    dic_new_boxes_conf = {}

    idx_dic = 0

    for element in dic_boxes.values():
        if element != None:
            box1, box2, conf = element
            dic_result_final[idx_dic] = list_class_norm[idx_dic][0]
            dic_new_boxes_conf[idx_dic] = [get_union_box(box1, box2),conf]
        else:
            # NO PASA
            dic_result_final[idx_dic] = None
            dic_new_boxes_conf[idx_dic] = None
        idx_dic += 1
    
    return dic_result_final, dic_new_boxes_conf

# def check_labels_match(result_predictions, true_labels):

def fill_with_opposite_elements(smaller_list, larger_list):
    # Define el elemento contrario
    opposite_element = {0:1, 2: 1, 1: 2}  # Puedes ajustar esto según tus necesidades

    # Llena la primera lista hasta que ambas listas tengan la misma longitud
    while len(smaller_list) < len(larger_list):
        for element in larger_list:
            if len(smaller_list) >= len(larger_list):
                break
            if element in opposite_element:
                smaller_list.append(opposite_element[element])
    
    return smaller_list

def fuse_predictions(result_predictions, list_labels_real):
    list_labels_pred = [value for value in result_predictions.values() if value is not None]

    amount_pred = len(list_labels_pred)

    amount_labels_txt = len(list_labels_real)

    # Caso que no haya encontrado todos los labels

    if amount_pred < amount_labels_txt:
        # Me llena la lista con valores contrarios a los reales

        list_complete_pred = fill_with_opposite_elements(list_labels_pred, list_labels_real)
        return list_complete_pred, 0
        
    elif amount_pred == amount_labels_txt:
        return list_labels_pred, 0

    else:
        # Devuelvo lista con la misma cantidad de labels y otra lista con los extras encontrados
        return list_labels_pred[:amount_labels_txt], list_labels_pred[amount_labels_txt:]

def labels_predictions(model_normal, model_termal, path_yaml_1, path_yaml_2, min_iou, distancia_max, thread_conf):
    images_normal, images_termal, list_txt = obtain_images_txt(path_yaml_1, path_yaml_2)
    
    y_true = []
    y_pred = []

    more_labels = []

    for img_normal, img_termal, file_txt in zip(images_normal, images_termal, list_txt):

        list_labels = extract_first_column(file_txt)

        result, _ = obtain_predictions(model_normal, model_termal, img_normal, img_termal, min_iou, distancia_max, thread_conf)
        
        # Fusionar predicciones (esto depende de tu lógica de fusión)
        list_pred_final, extra_labels = fuse_predictions(result, list_labels)
        y_true += list_labels
        y_pred += list_pred_final

        if extra_labels != 0:

            more_labels += extra_labels

    return y_true, y_pred, more_labels

def agregate_labels(y_true, y_pred, more_labels):
    amount_more_labels = len(more_labels)

    for i in range(amount_more_labels):
        y_true.append(3)

    y_pred += more_labels

    return y_true, y_pred

def evaluate_model_late(model_normal, model_termal, path_yaml_1, path_yaml_2, min_iou, distancia_max, thread_conf, path_save):
    labels = [0, 1, 2]

    y_true, y_pred, more_labels = labels_predictions(model_normal, model_termal, path_yaml_1, path_yaml_2, min_iou, distancia_max, thread_conf)

    precision_per_class = precision_score(y_true, y_pred, average=None, labels=range(len(labels)))
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=range(len(labels)))
    accuracy = accuracy_score(y_true, y_pred)

    print("El accuracy global es:", accuracy)
    print("La presición global es:", sum(precision_per_class) / 3)
    print("El recall global es:", sum(recall_per_class) / 3)

    if len(more_labels) > 0:
        labels = [0, 1, 2, 3]
        y_true, y_pred = agregate_labels(y_true, y_pred, more_labels)

    plot_confusion_matrix(y_true, y_pred, labels, path_save, more_labels)


def find_latest_tunex_folder(start_dir):
    latest_num = -1
    latest_folder = None

    # Recorre las subcarpetas del directorio de inicio
    for root, dirs, files in os.walk(start_dir):
        for dir_name in dirs:
            # Verifica si el nombre de la carpeta sigue el patrón 'tune' seguido de un número
            if dir_name == 'tune':
                latest_folder = os.path.join(root, dir_name)
            elif dir_name.startswith('tune'):
                match = re.match(r'tune(\d+)', dir_name)  # Usa expresión regular para extraer el número
                if match:
                    num = int(match.group(1))  # Obtiene el número después de 'tune'
                    if num > latest_num:
                        latest_num = num
                        latest_folder = os.path.join(root, dir_name)
    
    return latest_folder

def copy_best_hyperparameters(latest_folder, dest_dir):
    # Verifica si el archivo 'best_hyperparameters.yaml' existe en la carpeta encontrada
    file_path = os.path.join(latest_folder, 'best_hyperparameters.yaml')
    if os.path.exists(file_path):
        # Copia el archivo al directorio de destino
        shutil.copy(file_path, dest_dir)
        print(f"Archivo copiado a {dest_dir}")
    else:
        print("No se encontró el archivo 'best_hyperparameters.yaml' en la carpeta.")

def read_yaml(path_yaml):
    with open(path_yaml, 'r') as file:
        data = yaml.safe_load(file)

    return data

def delete_all_folders(path):
    # Verifica si la ruta existe y es un directorio
    if os.path.exists(path) and os.path.isdir(path):
        # Obtiene todas las carpetas dentro de la ruta
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            # Verifica si es un directorio
            if os.path.isdir(folder_path):
                # Elimina el directorio y su contenido
                shutil.rmtree(folder_path)
                print(f"Carpeta eliminada: {folder_path}")
    else:
        print(f"La ruta {path} no es válida o no es un directorio.")

def create_model(model_used, BASE_DIR, paht_yaml, new_name_model):
    model = YOLO("src/yolo11s.pt")

    model.tune(data=paht_yaml, epochs=model_used.epochs_hiper, iterations=model_used.iterations_hiper, optimizer=model_used.optimizer_hiper, plots=False, save=True, val=True, project=BASE_DIR+"/models/hiper")

    latest_folder = find_latest_tunex_folder(BASE_DIR+"/models/hiper")

    copy_best_hyperparameters(latest_folder, "models/best_params.yaml")

    delete_all_folders(BASE_DIR+"/models/hiper")

    best_params = read_yaml("models/best_params.yaml")

    results = model.train(
                data=paht_yaml,
                epochs=model_used.epochs_train,
                batch=model_used.batch,
                imgsz=model_used.imgsz,
                device=model_used.device,
                name=new_name_model,
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
                copy_paste=best_params["copy_paste"]
                )

def train_model(model_used, BASE_DIR):
    if model_used.name_model == "yolov11_late_fusion":
        create_model(model_used, BASE_DIR, BASE_DIR+"/src/IMGS_l_f.yaml", model_used.name_model_normal)
        create_model(model_used, BASE_DIR, BASE_DIR+"/src/TERM_l_f.yaml", model_used.name_model_termal)
    
    elif model_used.name_model == "yolov11_base":
        create_model(model_used, BASE_DIR, BASE_DIR+"/src/BASE_yolov11.yaml", model_used.name)

    elif model_used.name_model == "yolov11_GST":
        create_model(model_used, BASE_DIR, BASE_DIR+"/src/IMG_GST.yaml", model_used.name)

    elif model_used.name_model == "yolov11_HST":
        create_model(model_used, BASE_DIR, BASE_DIR+"/src/IMG_HST.yaml", model_used.name)