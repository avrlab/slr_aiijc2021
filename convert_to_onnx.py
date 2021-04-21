import onnx
import keras2onnx
from model import get_model
import argparse

parser = argparse.ArgumentParser(description='Script for convert .h5 (keras) to .onnx')
parser.add_argument('--data_info_path', type=str, default='data_processing/data_paths.pkl',
                    help='Path to file with a list of all image paths')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for training')
parser.add_argument('--weight_h5', type=str, default='',
                    help='Path to file with .h5 weights')
parser.add_argument('--weight_onnx', type=str, default='test_model.onnx',
                    help='Path to file with .onnx weights')


def main(args):
    if args.weight_h5 == '':
        print("Not found --weight_path!")
        exit()

    onnx_model_name = args.weight_onnx

    model = get_model(width=224, height=224, depth=32)
    model.load_weights(args.weight_h5)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    onnx.save_model(onnx_model, onnx_model_name)
    print("Saved file:", args.weight_onnx)


if __name__ == "__main__":
    args_info = parser.parse_args()
    main(args_info)
