import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_images_from_directory(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        img_path = os.path.join(directory_path, filename)
        if img_path.lower().endswith('.png'):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                images.append([img, os.path.basename(img_path)])
            else:
                print(f"Failed to load image at {img_path}")
        else:
            print(f"Skipped {img_path}, not an image file")
    return images


def visualize_mask(image, enhanced_image):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Original Contours')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(122)
    plt.title('Enhanced Contours')
    plt.imshow(enhanced_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def enhance_mask(dir_masks, dir_origin_images, save_dir_path, approximation, visualize):

    loaded_images = load_images_from_directory(dir_masks)
    for i, [img, img_name] in enumerate(loaded_images):

        original_image = cv2.imread(os.path.join(dir_origin_images, img_name))

        _, image = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        origin_contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        origin_image_contours = cv2.drawContours(original_image, origin_contours, -1, (255, 0, 0), 2)

        kernel = np.ones((3, 3), dtype=np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        for _ in range(5):
            image = cv2.medianBlur(image, ksize=11)
            kernel = np.ones((3, 3), dtype=np.uint8)
            image = cv2.erode(image, kernel, iterations=1)
        image = 255. - image
        image = image.astype(np.uint8)

        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        min_area_threshold = 2000
        for contour in contours:
            if cv2.contourArea(contour) < min_area_threshold:
                continue

            if approximation:
                epsilon_factor = 0.001
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)

            filtered_contours.append(contour)

        image_contours = cv2.drawContours(original_image, filtered_contours, -1, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(save_dir_path, img_name), image_contours)

        if visualize:
            visualize_mask(origin_image_contours, image_contours)


def enhance_mask_handler(args):
    enhance_mask(args.dir_masks_path,
                 args.dir_origin_paths,
                 args.save_dir_path,
                 args.approximation,
                 args.visualization)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_masks_path",
                        help="Path to directory with predicted masks",
                        required=True)
    parser.add_argument("--dir_origin_paths",
                        help="Path to directory with origin images",
                        required=True)
    parser.add_argument("--save_dir_path",
                        help="Where to save enhanced masks",
                        required=True)
    parser.add_argument("--approximation",
                        help="Whether to approximate and smooth contours",
                        type=bool,
                        default=False)
    parser.add_argument("--visualization",
                        help="Whether to plot comparison of origin and enhanced masks",
                        type=bool,
                        default=False)

    parser.set_defaults(func=enhance_mask_handler)

    args = parser.parse_args()
    args.func(args)
