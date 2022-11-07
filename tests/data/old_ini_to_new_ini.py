#!python

import configparser
import sys
import os


def old_to_new(inputfilename: str, outputfilename: str):

    fin = configparser.ConfigParser()
    fout = configparser.ConfigParser()

    fin.read(inputfilename)

    fout['advanced'] = fin['config']
    fout['advanced']['filter2d_smooth_low'] = fin['image_rendering']['filter2d_smooth_low']
    fout['advanced']['filter2d_smooth_high'] = fin['image_rendering']['filter2d_smooth_high']
    fout['advanced']['contrast_lower_threshold'] = fin['view']['contrast_lower_threshold']
    fout['advanced']['contrast_upper_threshold'] = fin['view']['contrast_upper_threshold']
    fout['advanced']['renorm_constant'] = fin['view']['renorm_constant']
    fout['advanced']['cuts_contrast_p_offset'] = fin['view']['cuts_contrast_p_offset']

    fout['image_rendering'] = fin['image_rendering']

    fout['view'] = fin['view']
    fout['view']['q_index'] = fin['image_rendering']['q_index']
    fout['view']['p_index'] = fin['image_rendering']['p_index']

    fout['view']['yz_contrast_enabled'] = fin['view']['contrast.enabled']
    fout['view']['yz_contrast_min'] = fin['view']['contrast_min']
    fout['view']['yz_contrast_max'] = fin['view']['contrast_max']
    fout['view']['yz_contrast_auto_enabled'] = fin['view']['contrast_auto_refresh']
    fout['view']['xz_contrast_enabled'] = fin['view']['contrast.enabled']
    fout['view']['xz_contrast_min'] = fin['view']['contrast_min']
    fout['view']['xz_contrast_max'] = fin['view']['contrast_max']
    fout['view']['xz_contrast_auto_enabled'] = fin['view']['contrast_auto_refresh']
    fout['view']['filter2d_contrast_enabled'] = fin['view']['contrast.enabled']
    fout['view']['filter2d_contrast_min'] = fin['view']['contrast_min']
    fout['view']['filter2d_contrast_max'] = fin['view']['contrast_max']
    fout['view']['filter2d_contrast_auto_enabled'] = fin['view']['contrast_auto_refresh']

    fout['view']['xy_img_accu_level'] = fin['config']['accumulation_buffer_size']

    fout['composite'] = {}

    to_add = configparser.ConfigParser()
    to_add.read("to_add.ini")
    for section in to_add.sections():
        for option in to_add.options(section):
            fout[section][option] = to_add[section][option]

    to_remove = configparser.ConfigParser()
    to_remove.read("to_remove.ini")
    for section in to_remove.sections():
        for option in to_remove.options(section):
            fout.remove_option(section, option)

    to_change = configparser.ConfigParser()
    to_change.read("to_change.ini")
    for section in to_change.sections():
        for option in to_change.options(section):
            fout[section][to_change[section][option]] = fin[section][option]

    with open(outputfilename, 'w') as output:
        fout.write(output)


def all_translation():
    folders = [name for name in os.listdir(
        ".") if os.path.isdir(name) and name != "inputs"]

    for f in folders:
        try:
            filename = os.path.join(f, "holovibes.ini")
            old_to_new(filename, filename)
        except:
            print(f"Could not translate folder: {f}.")


if __name__ == '__main__':

    argv = sys.argv
    argc = len(argv)

    if argc > 3:
        exit(1)

    if argc == 1:
        all_translation()
    else:
        if argc == 2:
            fin, fout = argv[1], argv[1]
        elif argc == 3:
            fin, fout = argv[1], argv[2]

        old_to_new(fin, fout)
