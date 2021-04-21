import argparse
import os
import pickle

parser = argparse.ArgumentParser(description='Script for generate paths')
parser.add_argument('--local_dir', type=str, default='../../prepared_data_crop',
                    help='Directory where the dataset is located')
parser.add_argument('--data_info_path', type=str, default='data_paths.pkl',
                    help='Path to file with a list of all image paths')


def main(args):
    print("Start generate paths...")

    data_paths = {'test': [], 'train': [], 'val': []}
    for dir_path, _, filenames in os.walk(args.local_dir):
        if len(filenames) != 0:
            paths_images = []

            for f in sorted(filenames):
                file_path = os.path.abspath(os.path.join(dir_path, f))
                paths_images.append(file_path)

            label = int(os.path.split(os.path.split(dir_path)[0])[-1])
            part = os.path.split(os.path.split(os.path.split(dir_path)[0])[0])[-1]
            data_paths[part].append([paths_images, label])

    with open(args.data_info_path, 'wb') as f:
        pickle.dump(data_paths, f, pickle.HIGHEST_PROTOCOL)
    print("Path generation completed")


if __name__ == "__main__":
    args_info = parser.parse_args()
    main(args_info)
