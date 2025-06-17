# import os
# import cv2
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from tqdm import tqdm
#
# def main():
#     input_dir = 'data/frames'
#     output_dir = 'data/results_tracking'
#     os.makedirs(output_dir, exist_ok=True)
#
#     print("[INFO] Загружаем YOLOv8...")
#     model = YOLO('yolov8n.pt')
#
#     print("[INFO] Инициализируем DeepSORT трекер...")
#     tracker = DeepSort(max_age=30)
#
#     batch_size = 8  # можно увеличить, если позволяет память
#
#     # Список путей к кадрам
#     img_paths = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.endswith(('.jpg', '.png'))]
#
#     print("[INFO] Начинаем обработку кадров батчами...")
#
#     for i in tqdm(range(0, len(img_paths), batch_size)):
#         batch_paths = img_paths[i:i+batch_size]
#
#         # Загружаем все изображения батча в память (cv2.imread)
#         batch_imgs = [cv2.imread(p) for p in batch_paths]
#
#         # YOLO может принять список numpy-изображений, поэтому передаем так
#         results = model(batch_imgs, verbose=False)
#
#         for frame, result, img_path in zip(batch_imgs, results, batch_paths):
#             detections = []
#             for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
#                 if int(cls) != 0:
#                     continue  # Только люди
#                 x1, y1, x2, y2 = map(int, box)
#                 detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), 'person'))
#
#             tracks = tracker.update_tracks(detections, frame=frame)
#
#             for track in tracks:
#                 if not track.is_confirmed():
#                     continue
#                 track_id = track.track_id
#                 x1, y1, x2, y2 = map(int, track.to_ltrb())
#
#                 # Уменьшаем рамку
#                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#                 w, h = x2 - x1, y2 - y1
#                 scale = 0.8
#                 new_w, new_h = int(w * scale), int(h * scale)
#                 x1_new = max(0, cx - new_w // 2)
#                 y1_new = max(0, cy - new_h // 2)
#                 x2_new = min(frame.shape[1], cx + new_w // 2)
#                 y2_new = min(frame.shape[0], cy + new_h // 2)
#
#                 cv2.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 2)
#                 cv2.putText(frame, f'ID: {track_id}', (x1_new, y1_new - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#
#             out_path = os.path.join(output_dir, os.path.basename(img_path))
#             cv2.imwrite(out_path, frame)
#
#     print("[INFO] Обработка завершена, результаты в", output_dir)
#
# if __name__ == "__main__":
#     main()
