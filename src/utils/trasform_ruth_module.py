import importlib

def modularization_ruth(route):
    spec = importlib.util.spec_from_file_location("contain_info", route)
    contain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contain)
    return contain