#!/usr/bin/env python

"""
Convert a HOLO file to AVI, MP4 and raw
"""

import sys
import argparse
from os.path import isfile
import cv2
import numpy as np
import math

import holo

# Returns (input_path, output_path)
def parse_cli() -> (str, str, int):
    parser = argparse.ArgumentParser(description='Convert HOLO to AVI.')
    parser.add_argument('-i', '--input',
                        help='input file (HOLO file)', type=str, required=True)
    parser.add_argument('-o', '--output',
                        help='output file (AVI file)', type=str, required=True)
    parser.add_argument('--fps',
                        help='output FPS', type=int, required=False, default=20)
    args = parser.parse_args()
    return (args.input, args.output, args.fps)

def holo_to_video(input_path: str, output_path: str, fourcc: int, fps: int):
    print(f"Export {input_path} to {output_path} at {fps} FPS")

    holo_file = holo.HoloFileReader(input_path)
    avi_file = cv2.VideoWriter(output_path, fourcc, fps, (holo_file.width, holo_file.height), False)

    for i in range(holo_file.nb_images):
        frame = holo_file.get_frame()
        avi_file.write(np.array(frame).reshape((holo_file.width, holo_file.height)).astype('uint8'))
        progress = math.ceil((i + 1) / holo_file.nb_images * 100)
        print(f"\rProgress: {progress}%", end="")

    avi_file.release()
    holo_file.close()
    print("\nDone")

def holo_to_raw(input_path: str, output_path: str):
    print(f"Export {input_path} to {output_path}")

    holo_file = holo.HoloFileReader(input_path)
    raw_file = open(output_path, "wb")

    data = holo_file.get_all_frames()
    raw_file.write(data)

    raw_file.close()
    holo_file.close()

    print("Done")

def raw_to_holo(input_path: str, output_path: str):
    print(f"Export {input_path} to {output_path}")

    width = int(input("Width: "))
    height = int(input("Height: "))
    bytes_per_pixel = int(input("Bytes per pixel: "))
    nb_images = int(input("Number of images: "))
    size = width * height * bytes_per_pixel * nb_images;

    raw_file = open(input_path, "rb")
    raw_data = raw_file.read(size)

    header = (width, height, bytes_per_pixel, nb_images)
    holo_file = holo.HoloFileWriter(output_path, header, raw_data)
    holo_file.write()
    holo_file.close()

    print("Done")

if __name__ == '__main__':
    input_path, output_path, fps = parse_cli()

    if not isfile(input_path):
        print(f"Error: no such file {input_path}.", file=sys.stderr)
        quit()

    input_ext = input_path.split('.')[-1]
    output_ext = output_path.split('.')[-1]

    if input_ext == "raw" and output_ext == "holo":
        raw_to_holo(input_path, output_path)
    elif input_ext == "holo" and output_ext == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        holo_to_video(input_path, output_path, fourcc, fps)
    elif input_ext == "holo" and output_ext == "avi":
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        holo_to_video(input_path, output_path, fourcc, fps)
    elif input_ext == "holo" and output_ext == "raw":
        holo_to_raw(input_path, output_path)
    else:
        print(f"Error: .{input_ext} to .{output_ext} is not supported.", file=sys.stderr)
        quit()