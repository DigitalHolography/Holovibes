#!/usr/bin/env python

import os
import json

def add_or_remove_key_value_pairs(data, new_key_values):
    for keys, value in new_key_values.items():
        current = data
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        if value is None:
            current.pop(keys[-1], None)  # Remove the key if value is None
        else:
            current[keys[-1]] = value  # Otherwise, add or update the key-value pair

def process_directory(directory_path, new_key_values):
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name == 'holovibes.json':
                json_file_path = os.path.join(root, file_name)
                with open(json_file_path, 'r') as file:
                    data = json.load(file)
                add_or_remove_key_value_pairs(data, new_key_values)
                with open(json_file_path, 'w') as file:
                    json.dump(data, file, indent=4)

# Example usage:
directory_path = '.'  # Change this to the directory where your JSON files are located
new_key_values = {
    ("image_rendering", "input_filter"): {"enabled": False, "type": "None"}, 
    ("color_composite_image", "hsv", "slider_shift"): None,
    ("color_composite_image", "hsv", "h", "slider_shift"): {"max": 1,"min": 0},
    ("color_composite_image", "hsv", "h", "frame_index", "activated"): False,
}

process_directory(directory_path, new_key_values)
