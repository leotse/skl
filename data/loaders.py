from os.path import dirname, join

from data.parsers import parse_csv_data as parse

training_data_file = "training_data.csv"
validation_data_file = "validation_data.csv"
employee_file = "employee.csv"


def load_training_data():
    file_path = join(dirname(__file__), training_data_file)
    return _load_data(file_path)


def load_validation_data():
    file_path = join(dirname(__file__), validation_data_file)
    return _load_data(file_path)


def _load_data(file_path):
    data, _ = parse(file_path)
    return [item[3] for item in data], [item[1] for item in data]
