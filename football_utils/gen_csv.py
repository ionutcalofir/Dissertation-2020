import os
import argparse

def gen_csv(data_path, output_csv):
    cls_to_id = {cls: i for i, cls in enumerate(sorted(os.listdir(data_path)))}

    with open(output_csv, 'w') as f:
        for cls in sorted(os.listdir(data_path)):
            for vid in os.listdir(os.path.join(data_path, cls)):
                f.write('{} {}\n'.format(os.path.join(data_path, cls, vid), cls_to_id[cls]))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='YOUR_DATASET_FOLDER/train_256/')
    parser.add_argument('--output_csv', type=str, default='YOUR_DATASET_FOLDER/train.csv')

    return parser.parse_args()

def main():
    args = parse_args()

    gen_csv(args.data_path, args.output_csv)

if __name__ == '__main__':
    main()
