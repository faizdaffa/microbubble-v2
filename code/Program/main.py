from ultralytics import YOLO
import ultralytics
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)
# Load a model
model = YOLO("yolov8l.yaml")  # build a new model from scratch
model.to('cuda')
# Use the model

ultralytics.checks()

if __name__ == "__main__":
    results = model.train(data="D:\BAGAS\code\microbubble\code\Program\config.yaml", epochs=1000)  # train the model