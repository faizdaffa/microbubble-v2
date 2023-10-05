import os
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import pandas as pd  # Add this import for pandas

model = YOLO("best.pt")

def convert_to_mm(x):
    return x * 0.0051400    # 1 mm = 0.0051400 px

# Initialize an empty DataFrame to store the results
result_df = pd.DataFrame(columns=["Image", "r_mean", "r_min", "r_max", "A_mean", "A_min", "A_max"])

IMAGE_PATH_FOLDER = '../../../../../Data/images/train/'
for img in os.listdir(IMAGE_PATH_FOLDER):
    if img.endswith('.JPG'):
        img_path = IMAGE_PATH_FOLDER + img
        r_array = []
        results = model.predict(source=img_path, show=False, save=False)

        for i, result in enumerate(results[0].boxes, start=1):
            # X, Y, Width, Height
            width = result.xywh[0][2]
            height = result.xywh[0][3]

            width = convert_to_mm(width)
            height = convert_to_mm(height)

            if width <= height:
                r_array.append(width)
            if height <= width:
                r_array.append(height)

        r_mean = (sum(r_array) / len(r_array))
        r_min  = min(r_array)
        r_max  = max(r_array)
        A_mean = np.pi * (sum(r_array) / len(r_array))**2
        A_min  = np.pi * r_min ** 2
        A_max  = np.pi * r_max ** 2

        # Append the results to the DataFrame
        result_df = result_df._append({
            "Image": img,
            "r_mean": r_mean,
            "r_min": r_min,
            "r_max": r_max,
            "A_mean": A_mean,
            "A_min": A_min,
            "A_max": A_max
        }, ignore_index=True)

# Save the DataFrame to a CSV file
result_df.to_csv("Result_test_model.csv", index=False)

# Print the DataFrame for verification
print(result_df)