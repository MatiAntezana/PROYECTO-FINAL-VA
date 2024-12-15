import os
import logging

def define_n_log():
    log_directory = os.path.join("..", "PROYECTO_FINAL_VA", "logs")

    archivos = os.listdir(log_directory)

    archivos_con_formato = [archivo for archivo in archivos if archivo.startswith('loger_') and archivo.endswith('.log')]

    if not archivos_con_formato:
        return "loger_1.log"

    numeros = [int(nombre_archivo.split('_')[1].split('.')[0]) for nombre_archivo in archivos_con_formato]

    max_n = max(numeros)
    return f"loger_{max_n + 1}.log"