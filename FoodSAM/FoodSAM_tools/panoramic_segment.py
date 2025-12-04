import os
import csv
import json
import math
import random
import cv2
import numpy as np

from FoodSAM_tools.enhance_semantic_masks import predict_sam_label


# ---------------------- Utils ----------------------
def _norm(path: str) -> str:
    return os.path.normpath(path)

def _is_dir(p: str) -> bool:
    return os.path.isdir(_norm(p))

def _ensure_dir(p: str):
    os.makedirs(_norm(p), exist_ok=True)

def _safe_imread(path: str, flags=cv2.IMREAD_COLOR):
    p = _norm(path)
    img = cv2.imread(p, flags)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    return img

def _compute_bbox_from_mask(mask_bool: np.ndarray):
    """Return [x0, y0, w, h] for a boolean mask. If empty -> [0,0,0,0]."""
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]

def get_IoU(bbox_red, bbox_green):
    # bbox: [x0, y0, x1, y1] (ở code dưới ta chuyển w,h -> x1,y1 trước khi gọi)
    ix_min = max(bbox_red[0], bbox_green[0])
    iy_min = max(bbox_red[1], bbox_green[1])
    ix_max = min(bbox_red[2], bbox_green[2])
    iy_max = min(bbox_red[3], bbox_green[3])

    iw = max(ix_max - ix_min, 0.0)
    ih = max(iy_max - iy_min, 0.0)
    inters = iw * ih

    red_area = max(bbox_red[2] - bbox_red[0], 0.0) * max(bbox_red[3] - bbox_red[1], 0.0)
    green_area = max(bbox_green[2] - bbox_green[0], 0.0) * max(bbox_green[3] - bbox_green[1], 0.0)
    uni = red_area + green_area - inters
    if uni <= 0:
        return 0.0
    return inters / uni


# ---------------------- Metadata ----------------------
def _write_minimal_metadata(img_dir: str, masks_path_name="sam_mask/masks.npy", metadata_path="sam_metadata.csv"):
    """
    Nếu chưa có sam_metadata.csv, tự tạo tối thiểu dùng masks.npy:
    header: id,bbox_x0,bbox_y0,bbox_w,bbox_h
    và từng dòng id tương ứng với index trong masks.npy
    """
    masks_file = _norm(os.path.join(img_dir, masks_path_name))
    if not os.path.exists(masks_file):
        # Không có masks -> không thể sinh metadata
        return False

    sam_masks = np.load(masks_file)  # shape: (N, H, W) hoặc (N, H, W) bool
    meta_csv = _norm(os.path.join(img_dir, metadata_path))
    _ensure_dir(os.path.dirname(meta_csv))

    with open(meta_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "bbox_x0", "bbox_y0", "bbox_w", "bbox_h"])
        for i in range(sam_masks.shape[0]):
            mask = sam_masks[i].astype(bool)
            x0, y0, w, h = _compute_bbox_from_mask(mask)
            writer.writerow([str(i), f"{x0:.1f}", f"{y0:.1f}", f"{w:.1f}", f"{h:.1f}"])
    return True


# ---------------------- Background -> Category ----------------------
def background2category(data_folder,
                        thresholds=(0.5, 0.6),
                        method_dir='object_detection',
                        background_dir='sam_mask_label',
                        metadata_path='sam_metadata.csv',
                        masks_path_name='sam_mask/masks.npy',
                        save_dir='background2category'):
    """
    Với từng ảnh (thư mục con) trong data_folder:
      - Đảm bảo có metadata (tạo nếu thiếu từ masks.npy)
      - Với từng file JSON detection trong method_dir:
        -> mở từng file nhãn nền (trong background_dir)
        -> nếu IoU bbox(SAM) vs bbox(detect) >= threshold => gán category cho mask nền
        -> lưu kết quả vào save_dir/method_name/<background_name>/threshold-<th>.txt
    """
    data_folder = _norm(data_folder)
    picture_ids = os.listdir(data_folder)

    for threshold in thresholds:
        for img_id in picture_ids:
            if img_id == 'sam_process.log':
                continue
            img_dir = _norm(os.path.join(data_folder, img_id))
            if not _is_dir(img_dir):
                continue

            # 1) Đảm bảo metadata tồn tại (nếu thiếu -> sinh từ masks.npy)
            metadata_file = _norm(os.path.join(img_dir, metadata_path))
            if not os.path.exists(metadata_file):
                ok = _write_minimal_metadata(img_dir, masks_path_name=masks_path_name, metadata_path=metadata_path)
                if not ok:
                    # Không có masks -> bỏ qua ảnh này
                    continue

            # 2) Đọc metadata thành list dict (id -> bbox)
            meta = np.genfromtxt(metadata_file, delimiter=',', dtype=str)
            if meta.ndim == 1:  # chỉ header?
                continue
            meta_columns = meta[0]
            meta_data = []
            for i in range(1, meta.shape[0]):
                item = {}
                for j in range(meta.shape[1]):
                    item[meta_columns[j]] = meta[i][j]
                meta_data.append(item)

            # 3) Duyệt các phương pháp detection
            method_root = _norm(os.path.join(img_dir, method_dir))
            if not os.path.exists(method_root):
                continue
            methods = [m for m in os.listdir(method_root)]
            for method in methods:
                method_path = _norm(os.path.join(method_root, method))
                if not os.path.isfile(method_path):
                    continue
                with open(method_path, 'r', encoding="utf-8") as f:
                    boxs = json.load(f)  # list of {"bounding_box":[x0,y0,x1,y1], "category_id":..., "category_name":...}

                # 4) Duyệt các file nhãn nền
                bg_root = _norm(os.path.join(img_dir, background_dir))
                if not os.path.exists(bg_root):
                    continue
                backgrounds = [b for b in os.listdir(bg_root) if os.path.isfile(_norm(os.path.join(bg_root, b)))]

                for background in backgrounds:
                    background_path = _norm(os.path.join(bg_root, background))

                    # Đọc parts (CSV-like .txt)
                    parts = []
                    with open(background_path, 'r', encoding='utf-8') as f:
                        line = f.readline()
                        columns = [c.strip() for c in line.split(',')]
                        line = f.readline()
                        while line:
                            vals = [c.strip() for c in line.split(',')]
                            item = {columns[i]: vals[i] for i in range(min(len(columns), len(vals)))}
                            parts.append(item)
                            line = f.readline()

                    # 5) Với từng SAM mask nền -> tìm detection có IoU cao nhất, nếu >= th thì gán nhãn
                    for i in range(len(parts)):
                        if parts[i].get('category_name', '') == 'background':
                            # meta_data & parts cần cùng index i (như code gốc)
                            # Nếu không đủ dòng, skip
                            if i >= len(meta_data):
                                continue
                            # bbox từ metadata (w,h)
                            try:
                                x0 = float(meta_data[i]['bbox_x0']); y0 = float(meta_data[i]['bbox_y0'])
                                w  = float(meta_data[i]['bbox_w']);  h  = float(meta_data[i]['bbox_h'])
                            except Exception:
                                continue
                            # convert -> x1,y1
                            box = [x0, y0, x0 + w, y0 + h]

                            best_iou, index = 0.0, -1
                            for j, det in enumerate(boxs):
                                bb = det.get('bounding_box', None)
                                if not bb or len(bb) != 4:
                                    continue
                                temp = get_IoU(box, bb)
                                if temp >= best_iou:
                                    index = j
                                    best_iou = temp
                            if index >= 0 and best_iou >= threshold:
                                det = boxs[index]
                                # mapping: object class id → panoptic id = base 104 + det_id
                                parts[i]['category_name'] = str(det.get('category_name', 'unknown'))
                                det_id = int(det.get('category_id', -1))
                                if det_id >= 0:
                                    parts[i]['category_id'] = str(det_id + 104)

                    # 6) Lưu kết quả
                    temp_dir = _norm(os.path.join(img_dir, save_dir, os.path.splitext(method)[0], os.path.splitext(background)[0]))
                    _ensure_dir(temp_dir)
                    save_path = _norm(os.path.join(temp_dir, f"threshold-{threshold}.txt"))
                    with open(save_path, mode='w', encoding='utf-8', newline='') as handle:
                        handle.write(','.join(columns) + '\n')
                        for part in parts:
                            row = [str(part.get(c, "")) for c in columns]
                            handle.write(','.join(row) + '\n')


# ---------------------- Viz helpers ----------------------
def random_color():
    # màu sáng, dễ thấy
    r = random.randint(120, 255)
    g = random.randint(120, 255)
    b = random.randint(120, 255)
    return [b, g, r]

def _draw_boxes_on_mask(vis_img, boxes, info, color_list, color_list2, num_class):
    font = cv2.FONT_HERSHEY_SIMPLEX
    H, W = vis_img.shape[:2]
    for idx, coord in enumerate(boxes):
        x0, y0, w, h = coord[:4]
        x1, y1 = x0 + w, y0 + h
        x0i, y0i = max(int(x0), 0), max(int(y0), 0)
        x1i, y1i = min(int(x1), W-1), min(int(y1), H-1)
        if x1i <= x0i or y1i <= y0i:
            continue

        label = int(info[idx][0])
        color = (color_list[label].tolist()
                 if 0 <= label < num_class
                 else color_list2[min(label, len(color_list2)-1)].tolist())
        cv2.rectangle(vis_img, (x0i, y0i), (x1i, y1i), color, 2)
        category_name = str(info[idx][1])
        cv2.putText(vis_img, category_name, (x0i + 3, max(y0i + 12, 12)), font, 0.35, color, 1)
    return vis_img


# ---------------------- Panoramic pipeline ----------------------
def visualization_save(mask_rgb, save_path, boxes, info, color_list, color_list2, num_class):
    """
    mask_rgb: HxWx3 (0..255)
    boxes: Nx5 [x0, y0, w, h, area] (area sử dụng khi cần sort/filter ngoài)
    info:  Nx2 [(label, name), ...]
    """
    vis = mask_rgb.astype(np.uint8).copy()
    vis = _draw_boxes_on_mask(vis, boxes, info, color_list, color_list2, num_class)
    cv2.imwrite(_norm(save_path), vis)


def panoramic_segment(data_folder, category_txt, color_list_path, num_class=104,
                      area_thr=0.01, ratio_thr=0.5, top_k=80,
                      masks_path_name="sam_mask/masks.npy",
                      panoramic_mask_name='panoramic_vis.png',
                      instance_mask_name='instance_vis.png',
                      method_dir='object_detection',
                      metadata_path='sam_metadata.csv',
                      new_label_save_dir='background2category',
                      method='od_UniDet',
                      sam_mask_label_file_name='sam_mask_label.txt',
                      pred_mask_file_name='pred_mask.png',
                      sam_mask_label_file_dir='sam_mask_label'):
    """
    - predict_sam_label: gán category cho từng SAM mask (background/food class…)
    - background2category: dùng detection để đẩy 'background' → 'panoptic object' khi IoU đủ lớn
    - Tạo panoramic_vis.png (màu theo class), instance_vis.png
    """
    data_folder = _norm(data_folder)
    if os.path.isfile(data_folder):
        data_folder = os.path.splitext(data_folder)[0]

    # B1: Bản nhãn theo SAM (background/food)
    predict_sam_label([data_folder], category_txt,
                      masks_path_name, sam_mask_label_file_name,
                      pred_mask_file_name, sam_mask_label_file_dir)

    # B2: Nâng background thành object dựa trên detection
    background2category(data_folder,
                        method_dir=method_dir,
                        background_dir=sam_mask_label_file_dir,
                        metadata_path=metadata_path,
                        masks_path_name=masks_path_name,
                        save_dir=new_label_save_dir)

    # B3: Chuẩn bị màu
    color_list = np.load(color_list_path).astype(np.uint8)
    color_list2 = color_list[::-1].copy()
    color_list[0] = [238, 239, 20]  # background

    # B4: Duyệt từng ảnh/thư mục con
    skip_names = {'sam_process.log', 'FoodSAM', '.git', '__pycache__', 'ckpts', 'configs'}

    for img_folder in os.listdir(data_folder):
        if img_folder in skip_names:
            continue

        full_dir = _norm(os.path.join(data_folder, img_folder))
        if not _is_dir(full_dir):
            continue

        # Đường dẫn input & masks
        img_path = _norm(os.path.join(full_dir, 'input.jpg'))
        sam_masks_path = _norm(os.path.join(full_dir, masks_path_name))

        # BẮT BUỘC phải có masks.npy để chạy các bước sau
        if not os.path.exists(sam_masks_path):
            continue

        # Đọc ảnh gốc nếu có; nếu không có thì suy kích thước từ masks
        if os.path.exists(img_path):
            img = _safe_imread(img_path, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
        else:
            sam_masks_probe = np.load(sam_masks_path)
            h, w = sam_masks_probe.shape[1], sam_masks_probe.shape[2]
            del sam_masks_probe  # tránh giữ 2 bản trong RAM

        # Nạp SAM masks (bắt buộc để xử lý phía dưới)
        sam_masks = np.load(sam_masks_path)

        # ---- Panoramic (dựa trên background2category threshold=0.5) ----
        category_info_path = _norm(os.path.join(full_dir, new_label_save_dir, method, sam_mask_label_file_dir, 'threshold-0.5.txt'))
        if not os.path.exists(category_info_path):
            # nếu chưa có file threshold-0.5 -> bỏ qua panoramic
            pass
        else:
            with open(category_info_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) > 1:
                category_info = lines[1:]
                # sort theo area giảm dần, cắt top_k
                category_info = sorted(category_info, key=lambda x: float(x.strip().split(',')[4]), reverse=True)[:top_k]

                enhanced_mask = np.zeros((h, w, 3), dtype=np.uint8)
                box_cats = {}

                for info_line in category_info:
                    parts = [p.strip() for p in info_line.split(',')]
                    if len(parts) < 5:
                        continue
                    idx = int(parts[0])
                    label = int(parts[1])
                    cat_name = parts[2]
                    count_ratio = float(parts[3])
                    area = float(parts[4])

                    if (area < area_thr and label < num_class) or (count_ratio < ratio_thr):
                        continue

                    sam_mask = sam_masks[idx].astype(bool)
                    # check mềm (tránh sai số làm tròn)
                    frac = sam_mask.sum() / max(1, (sam_mask.shape[0] * sam_mask.shape[1]))
                    if abs(frac - area) > 5e-3:
                        # bỏ qua assert cứng
                        pass

                    if label == 0:
                        continue
                    elif label < num_class:
                        enhanced_mask[sam_mask] = random_color()
                    else:
                        # guard out-of-range
                        lid = min(label, len(color_list) - 1)
                        enhanced_mask[sam_mask] = color_list[lid]

                    if label != 0:
                        box_cats[str(idx)] = (label, cat_name, area)

                # boxes từ metadata
                boxes = []
                info = []
                csv_path = _norm(os.path.join(full_dir, metadata_path))
                if os.path.exists(csv_path):
                    with open(csv_path, newline='', encoding='utf-8') as f:
                        rows = list(csv.reader(f))
                    # Bỏ header nếu có
                    start = 1 if rows and rows[0] and rows[0][0] == 'id' else 0
                    for row in rows[start:]:
                        if not row:
                            continue
                        row_id = row[0]
                        if row_id in box_cats:
                            label = box_cats[row_id][0]
                            name = box_cats[row_id][1]
                            area = box_cats[row_id][2]
                            x0, y0, w0, h0 = map(float, row[1:5])
                            boxes.append([x0, y0, w0, h0, area])
                            info.append((label, name))

                boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)
                save_dir_pan = full_dir
                _ensure_dir(save_dir_pan)
                panoramic_out = _norm(os.path.join(save_dir_pan, panoramic_mask_name))
                visualization_save(enhanced_mask, panoramic_out, boxes, info, color_list, color_list2, num_class)

        # ---- Instance (từ sam_mask_label gốc) ----
        instance_category_info_path = _norm(os.path.join(full_dir, sam_mask_label_file_dir, 'sam_mask_label.txt'))
        if os.path.exists(instance_category_info_path):
            with open(instance_category_info_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) > 1:
                instance_category_info = lines[1:]
                instance_category_info = sorted(
                    instance_category_info, key=lambda x: float(x.strip().split(',')[4]), reverse=True
                )[:top_k]

                inst_mask = np.zeros((h, w, 3), dtype=np.uint8)
                box_cats_inst = {}
                for info_line in instance_category_info:
                    parts = [p.strip() for p in info_line.split(',')]
                    if len(parts) < 5:
                        continue
                    idx = int(parts[0])
                    label = int(parts[1])
                    cat_name = parts[2]
                    count_ratio = float(parts[3])
                    area = float(parts[4])

                    if (area < area_thr and label < num_class) or (count_ratio < ratio_thr):
                        continue

                    sam_mask = sam_masks[idx].astype(bool)
                    # check mềm
                    frac = sam_mask.sum() / max(1, (sam_mask.shape[0] * sam_mask.shape[1]))
                    if abs(frac - area) > 5e-3:
                        pass

                    if label == 0:
                        continue
                    elif label < num_class:
                        inst_mask[sam_mask] = random_color()
                    else:
                        lid = min(label, len(color_list) - 1)
                        inst_mask[sam_mask] = color_list[lid]

                    if label != 0:
                        box_cats_inst[str(idx)] = (label, cat_name, area)

                # boxes cho instance
                boxes = []
                info = []
                csv_path = _norm(os.path.join(full_dir, metadata_path))
                if os.path.exists(csv_path):
                    with open(csv_path, newline='', encoding='utf-8') as f:
                        rows = list(csv.reader(f))
                    start = 1 if rows and rows[0] and rows[0][0] == 'id' else 0
                    for row in rows[start:]:
                        if not row:
                            continue
                        row_id = row[0]
                        if row_id in box_cats_inst:
                            label = box_cats_inst[row_id][0]
                            name = box_cats_inst[row_id][1]
                            area = box_cats_inst[row_id][2]
                            x0, y0, w0, h0 = map(float, row[1:5])
                            boxes.append([x0, y0, w0, h0, area])
                            info.append((label, name))

                boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)
                instance_out = _norm(os.path.join(full_dir, instance_mask_name))
                visualization_save(inst_mask, instance_out, boxes, info, color_list, color_list2, num_class)
