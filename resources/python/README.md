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