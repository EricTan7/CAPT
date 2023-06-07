"""
Usage:
python convert_path.py --input-path path/to/yout/captions_p2_train.txt --convert-path path/to/your/data_dir
"""
import os
import argparse


def main(args):
    converted = dict()
    with open(args.input_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            imname = os.path.join(args.convert_path, line[0])
            converted[imname] = line[1]

    output_path = os.path.join(os.path.dirname(args.input_path), "cvt_captions_p2_train.txt")
    with open(output_path, 'w') as f:
        for key, item in converted.items():
            f.write(key+'\t'+item+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="/mnt/sdb/tanhao/recognition/caltech-101/captions_p2_train.txt", help="path to dataset")
    parser.add_argument("--convert-path", type=str, default="", help="output directory")
    args = parser.parse_args()
    main(args)


    # python datasets/extraction/convert_path.py --input-path /mnt/sdb/tanhao/recognition/fgvc_aircraft/captions_p2.txt --convert-path images /mnt/sdb/tanhao/recognition/caltech-101/101_ObjectCategories/