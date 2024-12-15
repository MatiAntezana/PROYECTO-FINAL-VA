import logging
import os
import random
import shutil
import yaml
from PIL import Image
import io
from shutil import copy2

def delete_file_folder(ruta_carpeta):
    if not os.path.exists(ruta_carpeta):
        print(f"La carpeta '{ruta_carpeta}' no existe.")
        return

    for archivo in os.listdir(ruta_carpeta):
        ruta_archivo = os.path.join(ruta_carpeta, archivo)

        if os.path.isfile(ruta_archivo) or os.path.islink(ruta_archivo):
            os.unlink(ruta_archivo)  # Elimina archivos o enlaces simbólicos
        elif os.path.isdir(ruta_archivo):
            shutil.rmtree(ruta_archivo)  # Elimina carpetas y su contenido

    print(f"Todo el contenido de la carpeta '{ruta_carpeta}' ha sido eliminado.")

def rename_img_txt(carpeta_origen, nombre_base):
    archivos = os.listdir(carpeta_origen)
    imagenes_renombradas = []
    rename_txt = []
    index = 1

    # Filtrar imágenes normales y térmicas
    imgs_normal = [f for f in archivos if f.lower().endswith('.jpg') and '_R' not in f]
    imgs_termal = [f for f in archivos if f.lower().endswith('.jpg') and '_R' in f]

    for imagen_normal in sorted(imgs_normal):  # Ordenar para consistencia
        try:
            # Leer imagen normal
            ruta_normal = os.path.join(carpeta_origen, imagen_normal)
            with open(ruta_normal, "rb") as img_file:
                imagen_normal_data = Image.open(io.BytesIO(img_file.read()))

            # Nuevo nombre para la imagen normal
            nuevo_nombre_normal = f"{nombre_base}_{str(index).zfill(4)}.jpg"

            # Buscar archivo TXT correspondiente
            ruta_txt_normal = os.path.splitext(ruta_normal)[0] + ".txt"
            archivo_txt_normal = ruta_txt_normal if os.path.exists(ruta_txt_normal) else None
            nuevo_txt_normal = f"{nombre_base}_{str(index).zfill(4)}.txt"

            # Buscar imagen térmica correspondiente
            numero_imagen = os.path.splitext(imagen_normal)[0].split('_')[-1]
            numero_termica = str(int(numero_imagen) - 1).zfill(4)
            imagen_termica = next(
                (f for f in imgs_termal if numero_termica in f), None
            )

            imagen_termica_data = None
            archivo_txt_termica = None
            nuevo_txt_termica = None

            if imagen_termica:
                ruta_termica = os.path.join(carpeta_origen, imagen_termica)
                with open(ruta_termica, "rb") as img_file:
                    imagen_termica_data = Image.open(io.BytesIO(img_file.read()))

                # Nuevo nombre para la imagen térmica
                nuevo_nombre_termica = f"{nombre_base}_{str(index).zfill(4)}_T.jpg"

                # Buscar archivo TXT correspondiente
                ruta_txt_termica = os.path.splitext(ruta_termica)[0] + ".txt"
                archivo_txt_termica = ruta_txt_termica if os.path.exists(ruta_txt_termica) else None
                nuevo_txt_termica = f"{nombre_base}_{str(index).zfill(4)}_T.txt"

            # Agregar datos a las listas
            imagenes_renombradas.append((imagen_normal_data, imagen_termica_data))
            rename_txt.append(
                ((nuevo_txt_normal, archivo_txt_normal), (nuevo_txt_termica, archivo_txt_termica))
            )

            index += 1

        except Exception as e:
            print(f"Error procesando {imagen_normal}: {e}")

    return imagenes_renombradas, rename_txt

def construct_file_yaml(output_path, train_path, valid_path, test_path, classes, BASE_DIR, name_yaml):
    data = {
    'path': BASE_DIR+f"/{output_path}",
    'train': BASE_DIR+f"/{output_path}"+train_path,
    'val': BASE_DIR+f"/{output_path}"+valid_path,
    'test': BASE_DIR+f"/{output_path}"+test_path,
    "names": classes
    }
    ruta_archivo = "src/"+name_yaml

    with open(ruta_archivo, 'w') as archivo:
        yaml.dump(data, archivo, default_flow_style=False, allow_unicode=True)


def split_dataset_images(images, txt_files, output_path, seed=341, train_ratio=0.7, valid_ratio=0.2):
    # Crear carpetas de salida
    for split in ["train", "valid", "test"]:
        for img_type in ["img_normal", "termales"]:
            os.makedirs(os.path.join(output_path, split, img_type, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_path, split, img_type, "labels"), exist_ok=True)

    # Mezclar datos
    random.seed(seed)
    data = list(zip(images, txt_files))
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    splits = {
        "train": data[:train_end],
        "valid": data[train_end:valid_end],
        "test": data[valid_end:]
    }

    # Guardar imágenes y archivos TXT en carpetas correspondientes
    for split, split_data in splits.items():
        for idx, ((img_normal, img_termal), (txt_normal, txt_termal)) in enumerate(split_data):
            # Generar nombres únicos
            base_name = f"{split}_{idx + 1}"
            normal_img_name = f"{base_name}.jpg"
            thermal_img_name = f"{base_name}_T.jpg"
            normal_txt_name = f"{base_name}.txt"
            thermal_txt_name = f"{base_name}_T.txt"

            # Rutas de salida
            img_normal_output = os.path.join(output_path, split, "img_normal", "images", normal_img_name)
            img_termal_output = os.path.join(output_path, split, "termales", "images", thermal_img_name)
            txt_normal_output = os.path.join(output_path, split, "img_normal", "labels", normal_txt_name)
            txt_termal_output = os.path.join(output_path, split, "termales", "labels", thermal_txt_name)

            # Guardar imágenes en disco
            img_normal.save(img_normal_output, format="JPEG")
            img_termal.save(img_termal_output, format="JPEG")

            # Copiar archivos TXT
            copy2(txt_normal[1], txt_normal_output)
            copy2(txt_termal[1], txt_termal_output)

    print("División del dataset completada.")

def split_dataset_one_images(images, txt_files, output_path, type_model, seed=341, train_ratio=0.7, valid_ratio=0.2):
    # Crear carpetas de salida
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, "labels"), exist_ok=True)

    # Mezclar datos
    random.seed(seed)
    data = list(zip(images, txt_files))
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    splits = {
        "train": data[:train_end],
        "valid": data[train_end:valid_end],
        "test": data[valid_end:]
    }

    # Guardar imágenes y archivos TXT en carpetas correspondientes
    for split, split_data in splits.items():
        for idx, ((img_normal, img_termal), (txt_normal, txt_termal)) in enumerate(split_data):
            # Generar nombres únicos
            base_name = f"{split}_{idx + 1}"
            normal_img_name = f"{base_name}.jpg"
            thermal_img_name = f"{base_name}_T.jpg"
            normal_txt_name = f"{base_name}.txt"
            thermal_txt_name = f"{base_name}_T.txt"

            # Rutas de salida para las imágenes
            img_normal_output = os.path.join(output_path, split, "images", normal_img_name)
            img_termal_output = os.path.join(output_path, split, "images", thermal_img_name)

            # Rutas de salida para los archivos TXT
            txt_normal_output = os.path.join(output_path, split, "labels", normal_txt_name)
            txt_termal_output = os.path.join(output_path, split, "labels", thermal_txt_name)

            # Guardar imágenes en disco
            img_normal.save(img_normal_output, format="JPEG")
            copy2(txt_normal[1], txt_normal_output)

            if type_model == "BASE":
                img_termal.save(img_termal_output, format="JPEG")
                copy2(txt_termal[1], txt_termal_output)

    print("División del dataset completada.")

def delete_file_yaml(carpeta):    
    # Listar todos los archivos en la carpeta
    archivos = os.listdir(carpeta)
    
    # Filtrar solo los archivos con extensión .yaml
    archivos_yaml = [archivo for archivo in archivos if archivo.lower().endswith('.yaml')]
    
    # Eliminar cada archivo .yaml encontrado
    for archivo in archivos_yaml:
        ruta_archivo = os.path.join(carpeta, archivo)
        os.remove(ruta_archivo)
        print(f"Archivo eliminado: {ruta_archivo}")
    
    logging.info("Se eliminaron los .yaml")

def dataloder_diff(model_used, data_used, BASE_DIR):
    param_sets = data_used.param_sets

    delete_file_folder(data_used.output_path)
    list_cow, list_txt_cow = rename_img_txt(data_used.dataset_cow, "Cow")
    list_Deer, list_txt_deer = rename_img_txt(data_used.dataset_deer, "Deer")
    list_Horse, list_txt_horse = rename_img_txt(data_used.dataset_horse,  "Horse")
    list_images = list_cow + list_Deer + list_Horse
    list_txt = list_txt_cow + list_txt_deer + list_txt_horse
    delete_file_yaml("src")

    if model_used.name_model == "yolov11_late_fusion":
        split_dataset_images(list_images, list_txt, data_used.output_path, seed=param_sets["seed"], train_ratio=param_sets["train"], valid_ratio=param_sets["valid"])
        construct_file_yaml(data_used.output_path, "/train/img_normal/images", "/valid/img_normal/images", "/test/img_normal/images", data_used.classes, BASE_DIR, "IMGS_l_f.yaml")
        construct_file_yaml(data_used.output_path, "/train/termales/images", "/valid/termales/images", "/test/termales/images", data_used.classes, BASE_DIR, "TERM_l_f.yaml")

    elif model_used.name_model == "yolov11_base":
        split_dataset_one_images(list_images, list_txt, data_used.output_path, type_model="BASE", seed=param_sets["seed"], train_ratio=param_sets["train"], valid_ratio=param_sets["valid"])
        construct_file_yaml(data_used.output_path, "/train", "/valid", "/test", data_used.classes, BASE_DIR, "BASE_yolov11.yaml")

    elif model_used.name_model == "yolov11_HST":
        split_dataset_one_images(list_images, list_txt, data_used.output_path, type_model="HST", seed=param_sets["seed"], train_ratio=param_sets["train"], valid_ratio=param_sets["valid"])
        construct_file_yaml(data_used.output_path, "/train", "/valid", "/test", data_used.classes, BASE_DIR, "IMG_HST.yaml")
    
    elif model_used.name_model == "yolov11_GST":
        split_dataset_one_images(list_images, list_txt, data_used.output_path, type_model="GST", seed=param_sets["seed"], train_ratio=param_sets["train"], valid_ratio=param_sets["valid"])
        construct_file_yaml(data_used.output_path, "/train", "/valid", "/test", data_used.classes, BASE_DIR, "IMG_GST.yaml")

def get_dataloader(model_used, data_used, BASE_DIR, data_create):
    if data_create:
        dataloder_diff(model_used, data_used, BASE_DIR)
        logging.info("Se dividió correctamente los datasets")

    logging.info("Se salteo la creación de nuevos sets")
