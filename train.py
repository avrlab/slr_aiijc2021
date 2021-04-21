from utils.data_loader import DataGenerator
from model import get_model
import os
import tensorflow as tf
import tqdm
import argparse

parser = argparse.ArgumentParser(description='Script for train model')
parser.add_argument('--weights_dir', type=str, default='weights',
                    help='Directory where weights will be saved')
parser.add_argument('--data_info_path', type=str, default='data_processing/data_paths.pkl',
                    help='Path to file with a list of all image paths')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for training')

optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


def loss(model, x, y, training):
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def main(args):
    if not os.path.exists(args.weights_dir):
        os.makedirs(args.weights_dir)

    data_info_path = os.path.abspath(args.data_info_path)
    train_batch_generator = DataGenerator(data_info_path, args.batch_size, 'train')
    val_batch_generator = DataGenerator(data_info_path, args.batch_size, 'val')
    train_batch_generator.shuffle_data()

    model = get_model(width=224, height=224, depth=32)

    for epoch in range(args.num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        val_loss_avg = tf.keras.metrics.Mean()
        val_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for x, y in tqdm.tqdm(train_batch_generator, desc="epoch: " + str(epoch + 1) + "/" + str(args.num_epochs)):
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, model(x, training=True))

        train_batch_generator.shuffle_data()
        print("\nEpoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                      epoch_loss_avg.result(),
                                                                      epoch_accuracy.result()))
        for x, y in val_batch_generator:
            y_ = model(x)
            loss_value = loss_object(y_true=y, y_pred=y_)

            val_loss_avg.update_state(loss_value)
            val_accuracy.update_state(y, model(x, training=False))

        print("Val {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                  val_loss_avg.result(),
                                                                  val_accuracy.result()))
        if epoch % 10 == 0:
            model.save_weights(
                os.path.join(args.weights_dir,
                             '{:04d}_model_{:02d}.h5'.format(epoch, int(100 * val_accuracy.result()))))


if __name__ == "__main__":
    args_info = parser.parse_args()
    main(args_info)
