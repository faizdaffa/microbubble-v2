import os
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import pandas as pd  # Add this import for pandas

model = YOLO(r"D:\BAGAS\code\microbubble-v2\code\Program\runs\detect\train2_yolov8m\weights\best.pt")

def convert_to_mm(x):
    return x * 0.0051400    # 1 mm = 0.0051400 px

# Initialize an empty DataFrame to store the results
result_df = {
    "Image"  :[], 
    "r_mean" :[], 
    "r_min"  :[], 
    "r_max"  :[], 
    "A_mean" :[], 
    "A_min"  :[], 
    "A_max" :[]
}
IMAGE_PATH_FOLDER = r"D:\\BAGAS\\code\\microbubble-v2\\code\\Data\\images\\train\\"
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

        r_mean = float(r_mean ) 
        r_min  = float(r_min  )
        r_max  = float(r_max  )
        A_mean = float(A_mean )
        A_min  = float(A_min  )
        A_max  = float(A_max  )

        # Append the results to the DataFrame
        result_df["Image"  ].append(img_path ) 
        result_df["r_mean" ].append(r_mean) 
        result_df["r_min"  ].append(r_min ) 
        result_df["r_max"  ].append(r_max ) 
        result_df["A_mean" ].append(A_mean) 
        result_df["A_min"  ].append(A_min ) 
        result_df["A_max"  ].append(A_max ) 

df = pd.DataFrame(result_df)
# Save the DataFrame to a CSV file
df.to_csv("Result_test_model_8m.csv", index=False)

# Print the DataFrame for verification
# print(result_df)