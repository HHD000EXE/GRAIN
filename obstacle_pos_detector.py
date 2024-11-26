import cv2
import numpy as np
import os
import csv

def find_purple_circle(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found")
        return

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of purple color in HSV
    lower_purple = np.array([130, 50, 0])  # Adjust these values based on your definition of purple
    upper_purple = np.array([180, 150, 140])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    radius_last = 25
    for contour in contours:
        # Approximate the contour to a circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        area = cv2.contourArea(contour)
        center_last = center
        if area > 900:  # You might need to adjust this threshold based on the size of the circle you expect
            # Check if the shape is a circle
            circularity = (4 * np.pi * area) / (cv2.arcLength(contour, True) ** 2)
            if 0.1 < circularity <= 2.0:  # These values are typical for circles, adjust as necessary
                print(f"Circle detected at center: {center} with radius: {radius}")
                return center, radius

    print("No purple circle detected.")
    try:
        center = center_last
    except:
        center = (240, 240)
    radius = radius_last
    return center, radius

# Example usage
folder_path = 'Test_data/train_images(RGB)'
name_path = 'Test_data/train_images(D)'
file_names = os.listdir(folder_path)
csv_name = os.listdir(name_path)


# Prepare CSV file with header
csv_filename = "label.csv"
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Center X", "Center Y"])
    index = 0
    for files in file_names:
        center, radius = find_purple_circle(folder_path + '/' + files)
        writer.writerow([csv_name[index], center[0]/400.0, center[1]/400.0])
        index += 1