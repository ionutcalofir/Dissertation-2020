import os
import argparse

def gen_list(data_path, output_list):
    with open(output_list, 'w') as f:
        for cls in sorted(os.listdir(data_path)):
            for vid in os.listdir(os.path.join(data_path, cls)):
                f.write('{}\n'.format(os.path.join(data_path, cls, vid)))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='YOUR_DATASET_FOLDER/train/')
    parser.add_argument('--output_list', type=str, default='YOUR_DATASET_FOLDER/train_list.txt')

    return parser.parse_args()

def main():
    args = parse_args()

    gen_list(args.data_path, args.output_list)

if __name__ == '__main__':
    main()
