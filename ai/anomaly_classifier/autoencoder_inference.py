import os
import argparse

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support
from skimage import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import cv2


def ssim_score(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    ssim_score = metrics.structural_similarity(image1_gray,
                                               image2_gray,
                                               data_range=image1_gray.max() - image1_gray.min())
    return ssim_score


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(
        f"Data Metrics:\nAccuracy: {accuracy}\nPrecision: {precision}\n"
        f"Recall: {recall}\nF1 Score: {f1_score}\n")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Non-Anomaly", "Predicted Anomaly"],
                yticklabels=["Actual Non-Anomaly", "Actual Anomaly"])
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

    plt.savefig('Confusion Matrix.jpg')
    plt.close()


def display_and_save_comparison(originals, reconstructed, file_name, n=5):
    n = min(n, originals.shape[0], reconstructed.shape[0])
    fig, axs = plt.subplots(2, n, figsize=(5*n, 2))

    for i in range(n):
        axs[0, i].imshow(originals[i])
        axs[0, i].axis('off')
        axs[1, i].imshow(reconstructed[i])
        axs[1, i].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(file_name)
    plt.show()


def load_data(image_paths, image_shape=(256, 256, 3)):
    num_images = len(image_paths)
    all_images = np.empty((num_images,) + image_shape, dtype='float32')

    for i, img_path in enumerate(tqdm(image_paths)):
        img = tf.keras.preprocessing.image.load_img(str(img_path), target_size=image_shape)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img /= 255.
        all_images[i] = img
    return all_images


def load_test_data(dir_path_anomaly, dir_path_non_anomaly, image_shape=(256, 256, 3)):
    anomaly_files = os.listdir(dir_path_anomaly)
    non_anomaly_files = os.listdir(dir_path_non_anomaly)

    anomaly_paths = []
    for file in anomaly_files:
        file_path = os.path.join(dir_path_anomaly, file)
        anomaly_paths.append(file_path)

    non_anomaly_paths = []
    for file in non_anomaly_files:
        file_path = os.path.join(dir_path_non_anomaly, file)
        non_anomaly_paths.append(file_path)

    anomaly_data = load_data(anomaly_paths, image_shape)
    non_anomaly_data = load_data(non_anomaly_paths, image_shape)
    return anomaly_data, non_anomaly_data, anomaly_files, non_anomaly_files


def test(test_data, classes, img_width, img_height, batch_size, model_path, threshold, save_dir_reconstructed):

    autoencoder = load_model(model_path)

    dir_non_anomaly = os.path.join(test_data, classes[0])
    dir_anomaly = os.path.join(test_data, classes[1])

    image_shape = (img_width, img_height, 3)
    test_anomaly_data, test_non_anomaly_data, test_anomaly_files, test_non_anomaly_files = load_test_data(dir_anomaly,
                                                                                                          dir_non_anomaly,
                                                                                                          image_shape)

    anomaly_reconstructed = autoencoder.predict(test_anomaly_data, batch_size=batch_size)
    non_anomaly_reconstructed = autoencoder.predict(test_non_anomaly_data, batch_size=batch_size)

    anomaly_folder = os.path.join(save_dir_reconstructed, 'anomaly_reconstructed')
    non_anomaly_folder = os.path.join(save_dir_reconstructed, 'non_anomaly_reconstructed')

    if not os.path.isdir(anomaly_folder):
        os.makedirs(anomaly_folder)
    if not os.path.isdir(non_anomaly_folder):
        os.makedirs(non_anomaly_folder)

    for i in range(len(test_anomaly_files)):
        plt.imsave(f'{anomaly_folder}/{test_anomaly_files[i]}', anomaly_reconstructed[i])
    for i in range(len(test_non_anomaly_files)):
        plt.imsave(f'{non_anomaly_folder}/{test_non_anomaly_files[i]}', non_anomaly_reconstructed[i])

    anomaly_loss = []
    for anomaly_img, anomaly_reconstructed_img in zip(test_anomaly_data, anomaly_reconstructed):
        loss = ssim_score(anomaly_img, anomaly_reconstructed_img)
        anomaly_loss.append(loss)

    non_anomaly_loss = []
    for non_anomaly_img, non_anomaly_reconstructed_img in zip(test_non_anomaly_data, non_anomaly_reconstructed):
        loss = ssim_score(non_anomaly_img, non_anomaly_reconstructed_img)
        non_anomaly_loss.append(loss)

    thresholded_anomaly_loss = [1 if loss > threshold else 0 for loss in anomaly_loss]
    thresholded_non_anomaly_loss = [1 if loss > threshold else 0 for loss in non_anomaly_loss]

    true_anomaly_loss = [1 for _ in range(len(anomaly_loss))]
    true_non_anomaly_loss = [0 for _ in range(len(non_anomaly_loss))]

    y_true = true_anomaly_loss + true_non_anomaly_loss
    y_pred = thresholded_anomaly_loss + thresholded_non_anomaly_loss

    calculate_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)


def test_handler(args):
    test(args.test_data,
         args.classes,
         args.img_width,
         args.img_height,
         args.batch_size,
         args.model_path,
         args.threshold,
         args.save_dir_reconstructed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data",
                        help="Path to testing data",
                        required=True)
    parser.add_argument("--classes",
                        nargs="+",
                        help="Subfolders for each class in testing data folder",
                        default=['non_anomaly_fields', 'anomaly_fields'])
    parser.add_argument("--img_width",
                        help="Image width.",
                        type=int,
                        default=256)
    parser.add_argument("--img_height",
                        help="Image height.",
                        type=int,
                        default=256)
    parser.add_argument("--batch_size",
                        help="Batch size for model training",
                        type=int,
                        default=32)
    parser.add_argument("--model_path",
                        help="Path to the model",
                        required=True)
    parser.add_argument("--threshold",
                        help="Threshold to classify tiles",
                        required=True)
    parser.add_argument("--save_dir_reconstructed",
                        help="Folder path to save the reconstructed images",
                        required=True)

    parser.set_defaults(func=test_handler)

    args = parser.parse_args()
    args.func(args)
