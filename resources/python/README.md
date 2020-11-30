# Holovibes Python scripts

## Requirements

1. Have `python3` installed
2. Install dependencies with `pip install -r requirements.txt`

## Usage

### convert_holo.py

| From  | To    | Command                               |
|-------|-------|---------------------------------------|
| .holo | .avi  | `python3 -i input.holo -o output.avi` |
| .holo | .mp4  | `python3 -i input.holo -o output.mp4` |
| .holo | .raw  | `python3 -i input.holo -o output.raw` |
| .raw  | .holo | `python3 -i input.raw -o output.holo` |

*Note: when creating a .holo from a .raw the program will prompt the user mandatory parameters: width, height, bytes per pixel and number of frames.*