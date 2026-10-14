import os
import pickle

def save_data(data, filename, folder=""):
    """
    folder example: ["parent", "child", "grandchild"]
    """
    code_directory = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(code_directory, *folder)

    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, filename + ".pickle")
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_data(filename, folder=None):
    """
    folder example: ["parent", "child", "grandchild"]
    """
    if folder is None:
        folder = []

    code_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        code_directory,
        *folder,
        filename + ".pickle"
    )

    with open(file_path, "rb") as file:
        return pickle.load(file)