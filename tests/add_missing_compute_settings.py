#!/usr/bin/env python

import os
import sys
import json
import holo

def add_or_remove_key_value_pairs(data, new_key_values, holo=False):
    for keys, value in new_key_values.items():
        current = data
        keys=list(keys)
        if holo:
            keys.insert(0, "compute_settings")
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        if value is None:
            current.pop(keys[-1], None)  # Remove the key if value is None
        else:
            current[keys[-1]] = value  # Otherwise, add or update the key-value pair

def process_directory(directory_path, new_key_values, json_file_name='holovibes.json', ref_file_name='ref.holo'):
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name == json_file_name:
                json_file_path = os.path.join(root, file_name)
                with open(json_file_path, 'r') as file:
                    data = json.load(file)
                add_or_remove_key_value_pairs(data, new_key_values)
                with open(json_file_path, 'w') as file:
                    json.dump(data, file, indent=4)
            if file_name == ref_file_name:
                ref_file_path = os.path.join(root, file_name)
                holofile = holo.HoloFile.from_file(ref_file_path)
                add_or_remove_key_value_pairs(holofile.footer, new_key_values, holo=True)
                holofile.to_file(ref_file_path)

# Example usage:
directory_path = '.'  # Change this to the directory where your JSON files are located
new_key_values = {
    ("image_rendering", "input_filter"): {"enabled": False, "type": "None"}, 
    ("color_composite_image", "hsv", "slider_shift"): None,
    ("color_composite_image", "hsv", "h", "slider_shift"): {"max": 1.0,"min": 0.0},
    ("color_composite_image", "hsv", "h", "frame_index", "activated"): False,
    ("color_composite_image", "hsv", "h", "blur"): None,
}

holo_header_version = 3
holo_header_size = 64
holo_header_padding_size = 35

json_file_name = sys.argv[1] if len(sys.argv) >= 2 else 'holovibes.json'
ref_file_name = sys.argv[2] if len(sys.argv) >= 3 else 'ref.holo'
process_directory(directory_path, new_key_values, json_file_name, ref_file_name)
