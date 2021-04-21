import argparse
from utils.data_loader import DataGenerator
import os
import tqdm
from model import get_model
import tensorflow as tf
import onnxruntime
import numpy as np

parser = argparse.ArgumentParser(description='Script for test .h5 (keras model) or .onnx model')
parser.add_argument('--data_info_path', type=str, default='data_processing/data_paths.pkl',
                    help='Path to file with a list of all image paths')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for training')
parser.add_argument('--weight_path', type=str, default='',
                    help='Path to file with weight')
parser.add_argument('--custom_input', default=False, action='store_true',
                    help='For onnx model only, if you use custom input size')


def main(args):
    onnx_mode = False
    data_info_path = os.path.abspath(args.data_info_path)
    test_batch_generator = DataGenerator(data_info_path, args.batch_size, 'test')
    model = get_model(width=224, height=224, depth=32)

    if args.weight_path.split(".")[-1] == 'h5':
        model.load_weights(args.weight_path)
    elif args.weight_path.split(".")[-1] == 'onnx':
        sess = onnxruntime.InferenceSession(args.weight_path)
        onnx_mode = True
    elif args.weight_path == '':
        print("Not found --weight_path!")
        exit()

    print("Load weights from folder:", args.weight_path)

    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for x, y in tqdm.tqdm(test_batch_generator, desc="test"):
        if onnx_mode:
            if not args.custom_input:
                x = x[..., np.newaxis]
            x = x if isinstance(x, list) else [x]
            feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
            y_ = sess.run(None, feed)[0]
        else:
            y_ = model(x, training=False)

        test_accuracy.update_state(y, y_)

    print("Test: Accuracy: {:.3%}".format(test_accuracy.result()))


if __name__ == "__main__":
    args_info = parser.parse_args()
    main(args_info)
