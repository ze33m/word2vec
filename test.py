import os
def get_last_dataset():
    file_names = []
    files = [os.path.join("datasets", f) for f in os.listdir("datasets") if os.path.isfile(os.path.join("datasets", f))]
    last_dataset = max(files, key=os.path.getmtime)
    return last_dataset
