import os
import tqdm
import time
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Script for cropping images')
parser.add_argument('--local_dir', type=str, default='../../prepared_data',
                    help='Directory where the dataset is located')
parser.add_argument('--output_dir', type=str, default='../../prepared_data_crop',
                    help='Directory where the result will be saved')
parser.add_argument('--output_height', type=int, default=224,
                    help='Height of output images')


def main(args):
    model_file = "data_processing/face_detection/opencv_face_detector_uint8.pb"
    config_file = "data_processing/face_detection/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    input_height = 0

    for dir_path, _, filenames in os.walk(args.local_dir):
        if filenames:
            first_frame = cv2.imread(os.path.join(dir_path, filenames[0]))
            input_height = first_frame.shape[0]
            break

    assert input_height != 0, 'Something wrong with data'
    print("Input data height:", input_height)

    aspect_ratio = [16, 9]
    width_new = int(input_height / aspect_ratio[1] * aspect_ratio[0])

    # Magic numbers for head position
    y1 = input_height // 4
    y2 = y1 + input_height // 7
    x1 = width_new // 2 - width_new // 20
    x2 = width_new // 2 + width_new // 20
    target_pts = np.float32([[x1, y1], [x2, y1], [x1 + ((x2 - x1) // 2), y2]])

    print("Start prepare data...")
    start = time.time()

    for dir_path, _, filenames in tqdm.tqdm(os.walk(args.local_dir), total=len(list(os.walk(args.local_dir))),
                                            desc="Cropping"):
        statistic_pts = []
        for f in filenames[::5]:
            frame = cv2.imread(os.path.join(dir_path, f))
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            if frame_width / aspect_ratio[0] != frame_height / aspect_ratio[1]:
                ratio = frame_height / aspect_ratio[1]
                correct_width = int(ratio * aspect_ratio[0])

                if correct_width > frame_width:
                    frame = cv2.copyMakeBorder(frame, 0, 0, int((correct_width - frame_width) // 2),
                                               int((correct_width - frame_width) // 2), borderType=cv2.BORDER_CONSTANT,
                                               value=0)

                frame_height = frame.shape[0]
                frame_width = frame.shape[1]

            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * frame_width)
                    y1 = int(detections[0, 0, i, 4] * frame_height)
                    x2 = int(detections[0, 0, i, 5] * frame_width)
                    y2 = int(detections[0, 0, i, 6] * frame_height)

                    input_pts = np.float32(
                        [[x1, y1 + (y2 - y1) // 2], [x2, y1 + (y2 - y1) // 2], [x1 + (x2 - x1) // 2, y2]])
                    statistic_pts.append(input_pts)
                    break

            median_pts = np.median(np.asarray(statistic_pts), axis=0)

        for f in filenames:
            frame = cv2.imread(os.path.join(dir_path, f))

            M = cv2.getAffineTransform(median_pts, target_pts)
            M[1, 1] = M[0, 0]  # Do not scale in width

            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            warp_frame = cv2.warpAffine(frame, M, (frame_width, frame_height))

            new_w = round(frame_width / (frame_height / args.output_height))
            warp_frame = cv2.resize(warp_frame, (new_w, args.output_height))
            warp_frame = warp_frame[:, new_w // 2 - args.output_height // 2: new_w // 2 + args.output_height // 2, :]

            if not os.path.exists(dir_path.replace(args.local_dir, args.output_dir)):
                os.makedirs(dir_path.replace(args.local_dir, args.output_dir))
            cv2.imwrite(os.path.join(dir_path.replace(args.local_dir, args.output_dir), f), warp_frame)

    end = time.time()

    print("Data preparation completed: {:.1f} min".format((end - start) / 60))
    print("Saved in folder:", args.output_dir)


if __name__ == "__main__":
    args_info = parser.parse_args()
    main(args_info)
