import os
import cv2
import pickle
import scipy.io
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.io
from sklearn.metrics import precision_score, recall_score
import re



def log_results_to_file(filename, mota, idf1, IDFN, IDFP, IDSW, gt):
    with open(filename, "a") as file:  # Открываем файл в режиме добавления (a)
        file.write(f"=== Tracking Evaluation ===\n")
        file.write(f"MOTA: {mota}\n")
        file.write(f"IDF1: {idf1}\n")
        file.write(f"FN: {IDFN}\n")
        file.write(f"FP: {IDFP}\n")
        file.write(f"IDSW: {IDSW}\n")
        file.write(f"GT: {gt}\n")
        file.write(f"===========================\n\n")

def evaluate_1(gt_path, pred_path, iou_threshold=0.5):
    import pandas as pd

    gt = pd.read_csv(gt_path, header=None, usecols=[0, 1, 2, 3, 4, 5], names=['frame', 'id', 'x', 'y', 'w', 'h'])
    pred = pd.read_csv(pred_path, header=None, usecols=[0, 1, 2, 3, 4, 5], names=['frame', 'id', 'x', 'y', 'w', 'h'])

    frames = sorted(gt['frame'].unique())
    IDTP, IDFP, IDFN, IDSW = 0, 0, 0, 0
    prev_gt_to_pred = {}

    for frame in frames:
        gt_f = gt[gt['frame'] == frame]
        pr_f = pred[pred['frame'] == frame]

        matched_pred_ids = set()
        matched_gt_ids = set()
        current_gt_to_pred = {}

        for _, gt_row in gt_f.iterrows():
            gt_box = [gt_row['x'], gt_row['y'], gt_row['x'] + gt_row['w'], gt_row['y'] + gt_row['h']]
            gt_id = gt_row['id']
            best_iou = 0
            best_pred_id = None

            for _, pr_row in pr_f.iterrows():
                pred_id = pr_row['id']
                if pred_id in matched_pred_ids:
                    continue
                pr_box = [pr_row['x'], pr_row['y'], pr_row['x'] + pr_row['w'], pr_row['y'] + pr_row['h']]
                iou_score = iou(gt_box, pr_box)
                if iou_score > best_iou and iou_score >= iou_threshold:
                    best_iou = iou_score
                    best_pred_id = pred_id

            if best_pred_id is not None:
                IDTP += 1
                matched_pred_ids.add(best_pred_id)
                matched_gt_ids.add(gt_id)
                current_gt_to_pred[gt_id] = best_pred_id

                # Проверка ID switch
                if gt_id in prev_gt_to_pred:
                    if prev_gt_to_pred[gt_id] != best_pred_id:
                        IDSW += 1
            else:
                IDFN += 1

        IDFP += len(pr_f) - len(matched_pred_ids)
        prev_gt_to_pred = current_gt_to_pred

    mota = 1 - (IDFN + IDFP + IDSW) / len(gt) if len(gt) > 0 else 0.0
    idf1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN) if (2 * IDTP + IDFP + IDFN) > 0 else 0.0

    print("\n=== Tracking Evaluation (Manual) ===")
    print(f"MOTA:  {mota:.3f}")
    print(f"IDF1:  {idf1:.3f}")
    print(f"FN:    {IDFN}")
    print(f"FP:    {IDFP}")
    print(f"IDSW:  {IDSW}")
    print(f"GT:    {len(gt)}\n")

    log_results_to_file("tracking_evaluation.txt", mota, idf1, IDFN, IDFP, IDSW, len(gt))


def detect(model, max_frames=None):
    print("[INFO] Запускаем детекцию и аннотирование...")
    input_dir = 'data/frames'
    output_img_dir = 'data/results'
    cache_dir = 'data/detections_cache'
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    all_images = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))])

    if max_frames is not None:
        all_images = all_images[:max_frames]

    for img_name in all_images:

        path = os.path.join(input_dir, img_name)
        result = model(path)[0]  # пример с управлением порогами

        # Сохраняем аннотированное изображение
        annotated = result.plot()
        cv2.imwrite(os.path.join(output_img_dir, img_name), annotated)

        # Сохраняем боксы для последующего трекинга
        boxes_data = []
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                boxes_data.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), 'person'))

        with open(os.path.join(cache_dir, img_name + '.pkl'), 'wb') as f:
            pickle.dump(boxes_data, f)
        print(f"Кадр {img_name}: найдено {len(boxes_data)} людей")
        if boxes_data:
            print(f"Первые боксы: {boxes_data[:3]}")
    print(f"[INFO] Детекция завершена, обработано кадров: {len(all_images)}")




def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea)



import cv2
import numpy as np
import os
import pickle
import scipy.io

def evaluate(cache_dir='data/detections_cache',
             gt_path='data/annotations/mall_gt.mat',
             input_dir='data/frames',
             vis_dir='data/eval_vis',  # папка для сохранения визуализаций
             padding=0):

    print("[INFO] Запускаем оценку (по вхождению GT-точек в расширенные боксы) и количеству людей...")

    gt_data = scipy.io.loadmat(gt_path)
    frame_annotations = gt_data['frame'][0]
    counts = gt_data['count'].flatten()

    os.makedirs(vis_dir, exist_ok=True)

    TP_total, FP_total, FN_total = 0, 0, 0
    abs_count_errors = []

    img_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')])

    for i, img_name in enumerate(img_files):
        gt_points = frame_annotations[i][0][0]['loc']
        gt_count = int(counts[i])

        pkl_path = os.path.join(cache_dir, img_name + '.pkl')
        img_path = os.path.join(input_dir, img_name)

        if not os.path.exists(pkl_path):
            print(f"[WARNING] Нет данных детекции для {img_name}, пропускаем.")
            continue

        with open(pkl_path, 'rb') as f:
            pred_boxes = pickle.load(f)

        # Читаем изображение
        img = cv2.imread(img_path)

        # Визуализация GT точек
        for (gx, gy) in gt_points:
            cv2.circle(img, (int(gx), int(gy)), 4, (0, 255, 0), -1)  # зелёный кружок

        # Визуализация предсказанных боксов
        for (x, y, w, h), _, _ in pred_boxes:
            x_exp = x - padding
            y_exp = y - padding
            w_exp = w + 2 * padding
            h_exp = h + 2 * padding
            cv2.rectangle(img, (int(x_exp), int(y_exp)), (int(x_exp + w_exp), int(y_exp + h_exp)), (0, 0, 255), 2)  # красный прямоугольник

        # Сохраняем изображение
        cv2.imwrite(os.path.join(vis_dir, img_name), img)

        # Оценка
        gt_matched = set()
        pred_matched = set()

        for gi, (gx, gy) in enumerate(gt_points):
            for pi, ((x, y, w, h), _, _) in enumerate(pred_boxes):
                if pi in pred_matched:
                    continue

                x_exp = x - padding
                y_exp = y - padding
                w_exp = w + 2 * padding
                h_exp = h + 2 * padding

                if x_exp <= gx <= x_exp + w_exp and y_exp <= gy <= y_exp + h_exp:
                    TP_total += 1
                    gt_matched.add(gi)
                    pred_matched.add(pi)
                    break

        FP_total += len(pred_boxes) - len(pred_matched)
        FN_total += len(gt_points) - len(gt_matched)

        pred_count = len(pred_boxes)
        abs_count_errors.append(abs(pred_count - gt_count))

    precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
    recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_abs_count_error = np.mean(abs_count_errors) if abs_count_errors else 0

    print(f"[INFO] Precision: {precision:.3f}")
    print(f"[INFO] Recall: {recall:.3f}")
    print(f"[INFO] F1-score: {f1_score:.3f}")
    print(f"[INFO] Средняя абсолютная ошибка по количеству людей (MAE): {mean_abs_count_error:.3f}")
    print(f"[INFO] Визуализации сохранены в {vis_dir}")
    print("[INFO] Оценка завершена.")




def tracking(params,draw_trails):
    # if params is True:
    #     params = get_deepsort_params_from_user()
    #
    # tracker = DeepSort(
    #     max_age=params["max_age"],
    #     n_init=params["n_init"],
    #     max_cosine_distance=params["max_cosine_distance"],
    #     nn_budget=params["nn_budget"],
    #     embedder=params["embedder"],
    #     half=params["half"]
    # )
    # ... дальше твой трекинг-код
    print("[INFO] Запускаем трекинг по кешированным данным...")
    input_dir = 'data/frames'
    output_dir = 'data/results_tracking_1'
    cache_dir = 'data/detections_cache'
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Инициализируем DeepSORT трекер с улучшенными параметрами...")
    tracker = DeepSort(
        max_age=5,
        n_init=2,
        max_cosine_distance=0.9,
        nn_budget=250,
        embedder="mobilenet",
        half=False
    )

    max_frames = 30
    img_paths = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if f.endswith(('.jpg', '.png'))
    ][:max_frames]

    track_output_path = os.path.join(output_dir, "tracks.txt")
    track_file = open(track_output_path, "w")

    track_history = {}  # Для хранения траекторий

    for frame_idx, img_path in enumerate(tqdm(img_paths, desc="Tracking"), start=1):
        img_name = os.path.basename(img_path)
        cache_path = os.path.join(cache_dir, img_name + '.pkl')

        if not os.path.exists(cache_path):
            print(f"[WARN] Пропускаем кадр {img_name} — нет кеша.")
            continue

        with open(cache_path, 'rb') as f:
            detections = pickle.load(f)

        # Фильтрация по уверенности
        #detections = [det for det in detections if det[1] > 0.5]
        frame = cv2.imread(img_path)
        frame = boost_contrast_clahe(frame)
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            w, h = x2 - x1, y2 - y1

            # MOT формат
            line = f"{frame_idx},{track_id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
            track_file.write(line)

            # Центр объекта
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Обновление траектории
            if draw_trails:
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 30:
                    track_history[track_id] = track_history[track_id][-30:]

                # Отрисовка линии траектории
                for i in range(1, len(track_history[track_id])):
                    pt1 = track_history[track_id][i - 1]
                    pt2 = track_history[track_id][i]
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Отрисовка бокса
            scale = 0.8
            new_w, new_h = int(w * scale), int(h * scale)
            x1_new = max(0, cx - new_w // 2)
            y1_new = max(0, cy - new_h // 2)
            x2_new = min(frame.shape[1], cx + new_w // 2)
            y2_new = min(frame.shape[0], cy + new_h // 2)

            cv2.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1_new, y1_new - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, frame)

    track_file.close()

    trackeval_path = os.path.join('data')
    os.makedirs(trackeval_path, exist_ok=True)

    final_output_path = os.path.join(trackeval_path, 'track.txt')

    # Копируем содержимое tracks.txt
    with open(track_output_path, 'r') as src_file, open(final_output_path, 'w') as dst_file:
        for line in src_file:
            dst_file.write(line)

    print(f"[INFO] Треки успешно сохранены в {final_output_path}")
    print("[INFO] Трекинг завершён, результаты сохранены в", output_dir)
def boost_contrast_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))  # Максимальный эффект
    l2 = clahe.apply(l)

    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def get_deepsort_params_from_user():
    print("=== Настройка параметров DeepSort ===")
    try:
        max_age = int(input("max_age [по умолчанию 10]: ") or 10)
        n_init = int(input("n_init [по умолчанию 4]: ") or 4)
        max_cosine_distance = float(input("max_cosine_distance [по умолчанию 0.4]: ") or 0.4)
        nn_budget = int(input("nn_budget [по умолчанию 100]: ") or 100)
        embedder = input("embedder (mobilenet, torchreid, etc.) [по умолчанию mobilenet]: ") or "mobilenet"
        half_input = input("Использовать half-precision (True/False) [по умолчанию True]: ") or "True"
        half = half_input.strip().lower() == "true"
    except ValueError:
        print("Ошибка ввода. Используются значения по умолчанию.")
        max_age = 10
        n_init = 4
        max_cosine_distance = 0.4
        nn_budget = 100
        embedder = "mobilenet"
        half = True

    return {
        "max_age": max_age,
        "n_init": n_init,
        "max_cosine_distance": max_cosine_distance,
        "nn_budget": nn_budget,
        "embedder": embedder,
        "half": half
    }


def main():
    print("[INFO] Загружаем YOLOv8 модель один раз...")
    model = YOLO('yolov8n.pt')

    while True:
        print("\nВыберите действие:")
        print("1 - Запустить детекцию и сохранить боксы")
        print("2 - Оценить точность детекции")
        print("3 - Трекинг (по сохранённым данным, с визуализацией следов)")
        print("4 - Трекинг (по сохранённым данным, без следов)")
        print("5 - Трекинг (оценка)")
        print("6 - Проверить качество модели (валидация)")
        print("0 - Выход")

        choice = input("Введите номер команды: ").strip()

        if choice == '1':
            detect(model, 30)
        elif choice == '2':
            evaluate()
        elif choice == '3':
            tracking(draw_trails=True)
        elif choice == '4':
            tracking(params=True, draw_trails=False)
        elif choice == '5':
            evaluate_1("data/gt.txt", "data/track.txt")
        elif choice == '6':
            dataset_yaml = "F:\Python_Solutions\Жигалов\people_detector_yolo\mydataset.yaml"
            print("[INFO] Запускаем валидацию модели на вашем датасете...")
            metrics = model.val(data=dataset_yaml)
            print(f"mAP@0.5: {metrics.box.map50:.4f}")
            print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
            print(f"mAP@0.75: {metrics.box.map75:.4f}")
            print(f"mAP по категориям: {metrics.box.maps}")
        elif choice == '0':
            print("Выход...")
            break
        else:
            print("Неверный выбор, попробуйте снова.")

if __name__ == "__main__":
    main()

