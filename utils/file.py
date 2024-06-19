import os


def make_sure_dir_exists(dir_name):
    if not os.path.exists(f"{dir_name}"):
        try:
            os.makedirs(f"{dir_name}")
            print(f"Directory '{dir_name}' created successfully.")
        except OSError as e:
            print(f"Error: Failed to create directory '{dir_name}'. {e}")
    else:
            print(f"Directory '{dir_name}' already exists.")