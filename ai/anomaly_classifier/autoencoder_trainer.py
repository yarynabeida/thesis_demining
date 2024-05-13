import os
import argparse

import tensorflow as tf
from tensorflow.keras import layers, Input, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def build_encoder(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='linear')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    latent = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    encoder = models.Model(inputs, latent, name="encoder")

    encoder.summary()

    return encoder


def build_decoder(input_shape=(8, 8, 512)):

    inputs = layers.Input(shape=input_shape)
    x = layers.UpSampling2D(size=(2, 2))(inputs)

    x = layers.Conv2DTranspose(256, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(128, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(16, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    decoded = layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)

    decoder = models.Model(inputs, decoded, name="decoder")
    decoder.summary()

    return decoder


def build_autoencoder(input_shape=(256, 256, 3)):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(input_shape)

    inputs = layers.Input(shape=input_shape)
    encoded = encoder(inputs)
    decoded = decoder(encoded)

    autoencoder = models.Model(inputs, decoded, name="autoencoder")
    return autoencoder


def load_data(image_paths, image_shape=(256, 256, 3)):
    num_images = len(image_paths)
    all_images = np.empty((num_images,) + image_shape, dtype='float32')

    for i, img_path in enumerate(tqdm(image_paths)):
        img = tf.keras.preprocessing.image.load_img(str(img_path), target_size=image_shape)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img /= 255.
        all_images[i] = img
    return all_images


def load_train_data(data_directory_path, image_shape):
    all_files = os.listdir(data_directory_path)

    file_paths, labels = [], []
    directories = [directory for directory in all_files if os.path.isdir(os.path.join(data_directory_path, directory))]

    for directory in directories:
        tiles_dir_path = os.path.join(data_directory_path, directory + '/tiles')
        polygon_labels = os.listdir(tiles_dir_path)

        for polygon_label in polygon_labels:
            tile_files = os.listdir(os.path.join(tiles_dir_path, polygon_label))
            for tile_file in tile_files:
                tile_file_path = os.path.join(tiles_dir_path, polygon_label, tile_file)
                file_paths.append(tile_file_path)
                labels.append(polygon_label)

    train_dataset = load_data(file_paths, image_shape)
    return train_dataset, file_paths


def plot_metrics(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('Training loss.png')
    plt.close()


def train(dir_path, img_width, img_height, learning_rate, loss, batch_size, epochs, save_dir):

    image_shape = (img_width, img_height, 3)
    data, files = load_train_data(dir_path, image_shape)

    train_data, val_data, train_files, val_files = train_test_split(data, files, test_size=0.2, random_state=42)

    autoencoder = build_autoencoder(image_shape)
    autoencoder.summary()

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_path = os.path.join(save_dir, "model_epoch_{epoch:02d}.hdf5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_best_only=False,
                                 save_weights_only=True,
                                 verbose=1,
                                 period=1)

    history = autoencoder.fit(train_data,
                              train_data,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(val_data, val_data),
                              shuffle=True,
                              callbacks=[checkpoint])

    plot_metrics(history)
    autoencoder.save(f'{save_dir}/TEST_autoencoder_model', save_format='tf')


def train_handler(args):
    train(args.train_data,
          args.img_width,
          args.img_height,
          args.lr,
          args.loss,
          args.batch_size,
          args.epochs,
          args.save_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",
                        help="Path to training data",
                        required=True)
    parser.add_argument("--img_width",
                        help="Image width.",
                        type=int,
                        default=256)
    parser.add_argument("--img_height",
                        help="Image height.",
                        type=int,
                        default=256)
    parser.add_argument("--lr",
                        help="Learning rate at start",
                        type=float,
                        default=1e-4)
    parser.add_argument("--loss",
                        help="Loss function to be used",
                        default='mse',
                        choices=['mse', 'mae'])
    parser.add_argument("--batch_size",
                        help="Batch size",
                        type=int,
                        default=2)
    parser.add_argument("--epochs",
                        help="Number of epochs to train for",
                        type=int,
                        default=10)
    parser.add_argument("--save_dir",
                        help="Where to save the trained model",
                        required=True)

    parser.set_defaults(func=train_handler)

    args = parser.parse_args()
    args.func(args)
