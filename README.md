# Классификация жестов Русского жестового языка
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mv0sxOHLlpK4FcYkUAKhxB79IcJvp6yT)

## Описание
Пример нейросети для распознавания жестов и дактильной азбуки русского жестового языка (РЖЯ) на основе видеофрагментов, где демонстрируются жесты, обозначающие буквы и слова.

## Установка
1. Склонируйте репозиторий:
   ```shell
   git clone https://github.com/avrlab/slr_aiijc2021.git 
   cd slr_aiijc2021
   ```
2. Установите необходимые зависимости:
   ```shell
    pip install -r requirements.txt
   ```
3. Загрузите датасет с [сайта конкурса](https://aiijc.com/ru/). Он должен иметь следующую структуру:
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
   
## Предобработка данных
1. Разбейте видео на отдельные кадры:
   ```shell
   python data_processing/convert_videos_to_images.py \
          --local_dir data \
          --output_dir prepared_data
   ```
   
2. Выровняйте и обрежьте кадры до квадрата:
   ```shell
   python data_processing/crop_data.py \
          --local_dir prepared_data \
          --output_dir prepared_data_crop
   ```
   
3. Сгенерируйте пути к файлам:
   ```shell
   python data_processing/generate_data_paths.py \
          --local_dir prepared_data_crop \
          --data_info_path data_paths.pkl
   ```
## Обучение нейросети
   ```shell
   python train.py \
       --data_info_path data_paths.pkl \
       --num_epochs 100 \
       --batch_size 4 \
       --weights_dir weights
   ```
## Тестовый скрипт
Для формата `.h5`:
   ```shell
   python test.py \
       --data_info_path data_paths.pkl \
       --batch_size 4 \
       --weight_path baseline_weights/baseline_model.h5
   ```

Аналогично для `.onnx`:
   ```shell
   python test.py \
       --data_info_path data_paths.pkl \
       --batch_size 4 \
       --weight_path baseline_weights/baseline_model.onnx
   ```

## Google Colab
Если у вас нет подходящих вычислительных мощностей для запуска этого проекта, то вы можете попробовать аналогичный скрипт в Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mv0sxOHLlpK4FcYkUAKhxB79IcJvp6yT)
