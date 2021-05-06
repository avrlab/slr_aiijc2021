# Classification of signs from Russian Sign Language (RSL)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fj3AzFtRhbaNGWRt3xU4zsRfdbLNGx4u)

Russian version: [README.md](docs/README.md)

## Description
An example of a neural network for gesture recognition and the dactyl alphabet of the Russian Sign Language (RSL) based on video fragments, which demonstrate gestures that denote letters and words.

## Dataset Downloads
* Yandex Disk: [downloading link](https://disk.yandex.ru/d/ATux-KiyU0NzIQ)
* Contest website: [downloading link](https://aiijc.com/ru/task/1064/)

## Installation
1. Clone this repository:
   ```shell
   git clone https://github.com/avrlab/slr_aiijc2021.git 
   cd slr_aiijc2021
   ```
2. Install required python packages:
   ```shell
    pip install -r requirements.txt
   ```
3. Upload a dataset archive. It should have this basic structure:
    ```
    data
    ├── test
    │   ├── 00
    │   └── ...
    ├── train
    │   ├── 00
    │   └── ...
    └── val
        ├── 00
        └── ...
    ```
   
## Data pre-processing
1. Run the script to break down the video into frames:
   ```shell
   python data_processing/convert_videos_to_images.py \
          --local_dir data \
          --output_dir prepared_data
   ```
   
2. Сrop the pictures to square and align the position of people in the video:
   ```shell
   python data_processing/crop_data.py \
          --local_dir prepared_data \
          --output_dir prepared_data_crop
   ```
   
3. Generat file paths:
   ```shell
   python data_processing/generate_data_paths.py \
          --local_dir prepared_data_crop \
          --data_info_path data_paths.pkl
   ```
## Neural network training
   ```shell
   python train.py \
       --data_info_path data_paths.pkl \
       --num_epochs 100 \
       --batch_size 4 \
       --weights_dir weights
   ```
## Test script
Run the script for Keras weights:
   ```shell
   python test.py \
       --data_info_path data_paths.pkl \
       --batch_size 4 \
       --weight_path baseline_weights/baseline_model.h5
   ```

Or for the ONNX model:
   ```shell
   python test.py \
       --data_info_path data_paths.pkl \
       --batch_size 4 \
       --weight_path baseline_weights/baseline_model.onnx
   ```

## Google Colab
If you don't have the right computing power to run this project, then you can try Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fj3AzFtRhbaNGWRt3xU4zsRfdbLNGx4u)