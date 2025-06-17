# from ultralytics import YOLO
# import scipy.io
# import os
#
# gt = scipy.io.loadmat('data/annotations/mall_gt.mat')
# counts = gt['count'].flatten()
#
# input_dir = 'data/frames'
# model = YOLO('yolov8n.pt')
# pred_counts = []
#
# for i, name in enumerate(sorted(os.listdir(input_dir))):
#     if not name.endswith('.jpg'):
#         continue
#     result = model(os.path.join(input_dir, name))[0]
#     # Только люди (class 0)
#     num_people = sum(1 for c in result.boxes.cls if int(c) == 0)
#     pred_counts.append(num_people)
#
# diffs = [abs(int(gt) - int(pred)) for gt, pred in zip(counts[:len(pred_counts)], pred_counts)]
#
# print(f'Средняя ошибка: {sum(diffs)/len(diffs):.2f}')
