# from ultralytics import YOLO
# import os
# import cv2
#
# def main():
#     input_dir = 'data/frames'
#     output_dir = 'data/results'
#     os.makedirs(output_dir, exist_ok=True)
#
#     model = YOLO('yolov8n.pt')
#
#     for img_name in sorted(os.listdir(input_dir)):
#         if not img_name.endswith(('.jpg', '.png')):
#             continue
#         path = os.path.join(input_dir, img_name)
#         result = model(path)[0]
#         annotated = result.plot()
#         cv2.imwrite(os.path.join(output_dir, img_name), annotated)
#
