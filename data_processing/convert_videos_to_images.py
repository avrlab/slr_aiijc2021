import subprocess
import os
import tqdm
import time
import argparse

parser = argparse.ArgumentParser(description='Script for converting videos to images')
parser.add_argument('--local_dir', type=str, default='../../data',
                    help='Directory where the dataset is located')
parser.add_argument('--output_dir', type=str, default='../../prepared_data',
                    help='Directory where the result will be saved')
parser.add_argument('--max_height', type=int, default=360,
                    help='If the height of the video is above this threshold it will be scaled down')
parser.add_argument('--fps', type=int, default=10,
                    help='How many frames from one second will be saved')


def main(args):
    print("Start prepare data...")
    start = time.time()

    for dir_path, _, filenames in tqdm.tqdm(os.walk(args.local_dir), total=len(list(os.walk(args.local_dir))),
                                            desc="Prepare data"):
        for f in filenames:
            new_dir = os.path.join(dir_path.replace(args.local_dir, args.output_dir), f.split(".")[-2])
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            dir_info = os.path.join(new_dir, '%04d.jpg')

            scale = 'scale=-1:' + str(args.max_height)
            subprocess.call(['ffmpeg', '-loglevel', 'error', '-i', os.path.join(dir_path, f), '-vf', scale,
                             '-r', str(args.fps), '-qscale:v', '2', dir_info])
    end = time.time()

    print("Data preparation completed: {:.1f} min".format((end - start) / 60))
    print("Saved in folder:", args.output_dir)


if __name__ == "__main__":
    args_info = parser.parse_args()
    main(args_info)
