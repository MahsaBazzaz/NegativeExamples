import glob
import os
import pdb
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import numpy as np
platform_chars_unique = sorted(list(["-","X", "}", "{", "<", ">", "[", "]", "Q", "S"]))
cave_chars_unique = sorted(list(["-","X", "}", "{"]))
cave_doors_chars_unique = sorted(list(["-","X", "}", "{", "b", "B"]))
cave_portals_chars_unique = sorted(list(["-","X", "0", "1", "2", "3"]))
vertical_chars_unique = sorted(list(["-","X", "}", "{"]))
slide_chars_unique = sorted(list(["-","X", "}", "{", "#"]))
sokoban_chars_unique = sorted(list(["-","X", "@", "#", "o"]))

def find_matching_file(pattern):
    matching_files = glob.glob(pattern)
    return matching_files
    
def get_positive(game):
    if game == "platform":
        int2char = dict(enumerate(platform_chars_unique))
    elif game == "cave":
        int2char = dict(enumerate(cave_chars_unique))
    elif game == "cave_doors":
        int2char = dict(enumerate(cave_doors_chars_unique))
    elif game == "cave_portal":
        int2char = dict(enumerate(cave_portals_chars_unique))
    elif game == "vertical":
        int2char = dict(enumerate(vertical_chars_unique))
    elif game == "slide":
        int2char = dict(enumerate(slide_chars_unique))
    elif game == "crates":
        int2char = dict(enumerate(sokoban_chars_unique))

    char2int = {ch: ii for ii, ch in int2char.items()}
    num_tiles = len(char2int)

    levels = []
    labels = []
    current_block = []

    # Check if the given path is a valid directory
    parent_dir_solvable = f"./TheGGLCTexts/{game}/solvable/texts"

    if not os.path.isdir(parent_dir_solvable):
        print(f"Error: {parent_dir_solvable} is not a valid directory.")
        return

    # Iterate through all directories and subdirectories
    for root, dirs, files in os.walk(parent_dir_solvable):
        # For each file in the current directory
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.rstrip('\n')
                    if not line.startswith("META"):
                        ncoded_line = [char2int[x] for x in line]
                        current_block.append(ncoded_line)
                current_block = np.array(current_block)
                levels.append(current_block)
                current_block = []
                labels.append(0)

    levels = np.eye(num_tiles, dtype='uint8')[levels]
    return levels, np.array(labels)

def get_negative(game):
    if game == "platform":
        int2char = dict(enumerate(platform_chars_unique))
    elif game == "cave":
        int2char = dict(enumerate(cave_chars_unique))
    elif game == "cave_doors":
        int2char = dict(enumerate(cave_doors_chars_unique))
    elif game == "cave_portal":
        int2char = dict(enumerate(cave_portals_chars_unique))
    elif game == "vertical":
        int2char = dict(enumerate(vertical_chars_unique))
    elif game == "slide":
        int2char = dict(enumerate(slide_chars_unique))
    elif game == "crates":
        int2char = dict(enumerate(sokoban_chars_unique))

    char2int = {ch: ii for ii, ch in int2char.items()}
    num_tiles = len(char2int)

    levels0, labels0 = get_unsolvable(game, char2int, num_tiles)
    if game == "platform":
        levels1, labels1 = get_unusuable(game, char2int, num_tiles)
        levels = np.concatenate((levels0, levels1))
        labels = np.concatenate((labels0, labels1))
    else:
        levels = levels0
        labels = labels0

    return levels, labels

def get_unsolvable(game, char2int, num_tiles):
    levels = []
    labels = []
    current_block = []

    parent_dir_unsolvable = f"./TheGGLCTexts/{game}/unsolvable/texts"

    if not os.path.isdir(parent_dir_unsolvable):
        print(f"Error: {parent_dir_unsolvable} is not a valid directory.")
        return

    # Iterate through all directories and subdirectories
    for root, dirs, files in os.walk(parent_dir_unsolvable):
        # For each file in the current directory
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.rstrip('\n')
                    if not line.startswith("META"):
                        ncoded_line = [char2int[x] for x in line]
                        current_block.append(ncoded_line)
                current_block = np.array(current_block)
                levels.append(current_block)
                current_block = []
                labels.append(1)

    levels = np.eye(num_tiles, dtype='uint8')[levels]
    return levels, np.array(labels)

def get_unusuable(game, char2int, num_tiles):
    levels = []
    labels = []
    current_block = []

    parent_dir_unsolvable = f"./TheGGLCTexts/{game}/unusable/texts"

    if not os.path.isdir(parent_dir_unsolvable):
        print(f"Error: {parent_dir_unsolvable} is not a valid directory.")
        return

    # Iterate through all directories and subdirectories
    for root, dirs, files in os.walk(parent_dir_unsolvable):
        # For each file in the current directory
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.rstrip('\n')
                    ncoded_line = [char2int[x] for x in line]
                    current_block.append(ncoded_line)

                current_block = np.array(current_block)
                levels.append(current_block)
                current_block = []
                labels.append(1)
    levels = np.eye(num_tiles, dtype='uint8')[levels]
    return levels, np.array(labels)

def make_arrays_equal_length(arr1, arr2):
    len1 = len(arr1)
    len2 = len(arr2)

    if len1 > len2:
        np.random.shuffle(arr1)  # Shuffle the bigger array
        arr1 = arr1[:len2]
    elif len2 > len1:
        np.random.shuffle(arr2)  # Shuffle the bigger array
        arr2 = arr2[:len1]
    return np.array(arr1), np.array(arr2)

def map_output_to_symbols(game, integers):
    if game == "platform":
        int2char = dict(enumerate(platform_chars_unique))
    elif game == "cave":
        int2char = dict(enumerate(cave_chars_unique))
    elif game == "cave_doors":
        int2char = dict(enumerate(cave_doors_chars_unique))
    elif game == "cave_portal":
        int2char = dict(enumerate(cave_portals_chars_unique))
    elif game == "vertical":
        int2char = dict(enumerate(vertical_chars_unique))
    elif game == "slide":
        int2char = dict(enumerate(slide_chars_unique))
    elif game == "crates":
        int2char = dict(enumerate(sokoban_chars_unique))
    return [[int2char[i.item()] for i in row] for row in integers]

def get_reach_move(game):
    if game == "platform":
        return "platform"
    elif game == "cave":
        return "maze"
    elif game == "cave_doors":
        return "maze"
    elif game == "cave_portal":
        return "maze"
    elif game == "vertical":
        return "supercat-new"
    elif game == "slide":
        return "tomb"
    elif game == "crates":
        return None
    
def get_cols_rows(game):
    if game == "platform":
        return 16,32
    elif game == "cave":
        return 16,32
    elif game == "cave_doors":
        return 16,16
    elif game == "cave_portal":
        return 16,16
    elif game == "vertical":
        return 20,16
    elif game == "slide":
        return 32,26
    elif game == "crates":
        return 16,16
    
def get_z_dims(game):
    if game == "platform":
        chars =platform_chars_unique
    elif game == "cave":
        chars = cave_chars_unique
    elif game == "cave_doors":
        chars = cave_doors_chars_unique
    elif game == "cave_portal":
        chars = cave_portals_chars_unique
    elif game == "vertical":
        chars = vertical_chars_unique
    elif game == "slide":
        chars = slide_chars_unique
    elif game == "crates":
        chars = sokoban_chars_unique
    return len(chars)