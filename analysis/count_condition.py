import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pandas as pd

def get_level(input):
    level = []
    with open(input, 'r') as file:
        for line in file:
            level.append(line.strip())
    return level

def count_mario_patterns(array):
    count = 0
    for row_idx in range(len(array) - 1):
        for col_idx in range(len(array[0]) - 1):
            if array[row_idx][col_idx:col_idx+2] == "<>" and array[row_idx+1][col_idx:col_idx+2] == "[]":
                count += 1
    return count

def count_cave_treasures_patterns(array):
    count = 0
    for row_idx in range(len(array) - 1):
        for col_idx in range(len(array[0]) - 1):
            if array[row_idx][col_idx] == "2":
                count += 1
    return count

def count_mario_incomplete_patterns(array):
    count = 0
    for row_idx in range(len(array) - 1):
        for col_idx in range(len(array[0]) - 1):
            if (array[row_idx][col_idx:col_idx+2] == "<>" and 
                (row_idx == len(array) - 1 or array[row_idx+1][col_idx:col_idx+2] != "[]")) or \
               ((array[row_idx] == "<" and array[row_idx][col_idx:col_idx+2] != ">") or 
                (array[row_idx] != "<" and array[row_idx][col_idx:col_idx+2] == ">")):
                count += 1
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    game = 'mario'
    ins = 1
    epoch = 3000
    batch_size = 1000
    model = "C"
    condition = 1
    
    playable_correct = 0
    correct = 0

    folder_pattern = f'./out/{game}/{epoch}/{ins}/{model}_{condition}'
    for j in range (0, batch_size):
        path_file = f"{folder_pattern}/{str(j)}.path.lvl"
        level_file = f"{folder_pattern}/{str(j)}.lvl"
        if os.path.exists(level_file):
            level = get_level(level_file)
            if game == "mario":
                pattern_counts = count_mario_patterns(level)
            if game == "cave_treasures":
                pattern_counts = count_cave_treasures_patterns(level)
            if pattern_counts == condition:
                if os.path.exists(path_file):
                    playable_correct += 1
                else:
                    correct += 1


    print("playable_correct: ", playable_correct)
    print("correct: ", correct)


    

