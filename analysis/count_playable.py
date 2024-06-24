import os

def count_path_files(directory):
    # Initialize a counter
    png_count = 0
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file ends with .png
        if filename.lower().endswith('.path.lvl'):
            png_count += 1
    
    # Print the result
    print(f"Number of .path.lvl files in '{directory}': {png_count}")

# Example usage:
if __name__ == "__main__":
    parent_directory = './out/artifacts/cave/1/5000/C'
    count_path_files(parent_directory)