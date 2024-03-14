# Holovibes Python scripts

## Requirements

*Note: It is recommended to move the `python/` folder from Holovibes installation directory (usually `C:/Program Files/Holovibes/X.X.X/`) to another place to avoid permissions issues.*

1. Have [python3](https://www.python.org/downloads/) installed
2. Install dependencies with `pip install -r requirements.txt`

## Usage

### convert_holo.py

| From  | To    | Command                                         |
|-------|-------|-------------------------------------------------|
| .holo | .avi  | `python3 convert_holo.py input.holo output.avi` |
| .holo | .mp4  | `python3 convert_holo.py input.holo output.mp4` |
| .holo | .raw  | `python3 convert_holo.py input.holo output.raw` |
| .raw  | .holo | `python3 convert_holo.py input.raw output.holo` |

For .avi and .mp4 you can specify the output video FPS (by default 20) with `--fps 30`.

*Note: when creating a .holo from a .raw the program will prompt the user mandatory parameters: width, height, bytes per pixel and number of frames.*

### add_missing_compute_settings.py

Used to add/remove compute_settings from .holo and .json when changes have been made in the code.

The script must be run in the tests folder :
```sh
Holovibes/tests>$ ./add_missing_compute_settings.py [json_name] [holo_name]
```
json_name and holo_name are the .json and .holo that the script will modify. If you don't specify the names, it will modify all the **holovibes.json** and **ref.holo** recursively in the current folder and subfolders.

In order to add and remove keys, modify the directory "new_key_values" in the file. Here is an example usage:
```py
directory_path = '.'  # Change this to the directory where your JSON files are located
new_key_values = {
    ("image_rendering", "input_filter"): {"enabled": False, "type": "None"}, # Adds a key
    ("color_composite_image", "hsv", "slider_shift"): None, # Removes the key
    ("color_composite_image", "hsv", "h", "slider_shift"): {"max": 1.0,"min": 0.0}, # Adds the key
    ("color_composite_image", "hsv", "h", "frame_index", "activated"): False, # Adds the key
    ("color_composite_image", "hsv", "h", "blur"): None, # Removes the key
}
```