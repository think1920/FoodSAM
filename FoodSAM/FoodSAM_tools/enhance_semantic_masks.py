import os
import cv2
import logging
import numpy as np


# ---------- Helpers ----------
def _read_mask_channel_r(path: str) -> np.ndarray:
    """Read PNG mask robustly: return R channel if 3-ch, else single channel."""
    p = os.path.normpath(path)
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {p}")
    if img.ndim == 3:
        return img[:, :, 2]
    return img


# ---------- Stage 1: Assign category to each SAM mask ----------
def calculate_single_image_masks_label(
    mask_file: str,
    pred_mask_file: str,
    category_list: list,
    sam_mask_label_file_name: str,
    sam_mask_label_file_dir: str,
):
    """
    Write per-SAM-mask category:
        id,category_id,category_name,category_count_ratio,mask_count_ratio
    """
    sam_mask_data = np.load(os.path.normpath(mask_file))
    pred_mask_img = _read_mask_channel_r(pred_mask_file)

    H, W = pred_mask_img.shape[:2]
    shape_size = H * W
    logger = logging.getLogger()

    folder_path = os.path.dirname(os.path.normpath(pred_mask_file))
    sam_mask_category_folder = os.path.join(folder_path, sam_mask_label_file_dir)
    os.makedirs(sam_mask_category_folder, exist_ok=True)
    mask_category_path = os.path.join(sam_mask_category_folder, sam_mask_label_file_name)

    with open(mask_category_path, "w", encoding="utf-8") as f:
        f.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")

        for i in range(sam_mask_data.shape[0]):
            single_mask = sam_mask_data[i].astype(bool)
            if not single_mask.any():
                continue

            single_mask_labels = pred_mask_img[single_mask]
            if single_mask_labels.size == 0:
                continue

            unique_values, counts = np.unique(
                single_mask_labels, return_counts=True, axis=0
            )
            max_idx = np.argmax(counts)
            single_mask_category_label = int(unique_values[max_idx])
            count_ratio = counts[max_idx] / counts.sum()

            # Bound-check category name
            if 0 <= single_mask_category_label < len(category_list):
                cat_name = category_list[single_mask_category_label]
            else:
                cat_name = "unknown"

            logger.info(
                f"{folder_path}/sam_mask/{i} assign label: "
                f"[ {single_mask_category_label}, {cat_name}, {count_ratio:.2f}, {counts.sum()/shape_size:.4f} ]"
            )
            f.write(
                f"{i},{single_mask_category_label},{cat_name},{count_ratio:.2f},{counts.sum()/shape_size:.4f}\n"
            )


def predict_sam_label(
    data_folder,
    category_txt,
    masks_path_name="sam_mask/masks.npy",
    sam_mask_label_file_name="sam_mask_label.txt",
    pred_mask_file_name="pred_mask.png",
    sam_mask_label_file_dir="sam_mask_label",
):
    """Walk data_folder/* and write per-SAM-mask categories if masks + pred_mask exist."""
    # Load categories
    category_lists = []
    with open(category_txt, "r", encoding="utf-8") as f:
        category_lines = f.readlines()
        category_list = [
            " ".join(line_data.split("\t")[1:]).strip() for line_data in category_lines
        ]
        category_lists.append(category_list)

    # data_folder is list in caller
    for test_path, category_list in zip(data_folder, category_lists):
        for img_id in os.listdir(test_path):
            img_dir = os.path.join(test_path, img_id)
            if not os.path.isdir(img_dir):
                continue

            mask_file_path = os.path.join(img_dir, masks_path_name)
            pred_mask_file_path = os.path.join(img_dir, pred_mask_file_name)
            mask_file_path = os.path.normpath(mask_file_path)
            pred_mask_file_path = os.path.normpath(pred_mask_file_path)

            if os.path.exists(mask_file_path) and os.path.exists(pred_mask_file_path):
                calculate_single_image_masks_label(
                    mask_file_path,
                    pred_mask_file_path,
                    category_list,
                    sam_mask_label_file_name,
                    sam_mask_label_file_dir,
                )


# ---------- Stage 2: Visualize enhanced mask ----------
def visualization_save(mask: np.ndarray, save_path: str, img_path: str, color_list: np.ndarray):
    values = set(mask.flatten().tolist())
    if not values:
        return

    img_path = os.path.normpath(img_path)
    base = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(f"Cannot read base image for visualization: {img_path}")

    h, w = base.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8)

    max_idx = len(color_list) - 1
    for v in values:
        if v < 0:
            continue
        idx = int(v) if v <= max_idx else 0
        m = (mask == v)
        result[m] = color_list[idx]

    vis = cv2.addWeighted(base, 0.5, result, 0.5, 0)
    cv2.imwrite(os.path.normpath(save_path), vis)


# ---------- Stage 3: Enhance masks ----------
def enhance_masks(
    data_folder,
    category_txt,
    color_list_path,
    num_class=104,
    area_thr=0,
    ratio_thr=0.5,
    top_k=80,
    masks_path_name="sam_mask/masks.npy",
    new_mask_label_file_name="semantic_masks_category.txt",
    pred_mask_file_name="pred_mask.png",
    enhance_mask_name="enhance_mask.png",
    enhance_mask_vis_name="enhance_vis.png",
    sam_mask_label_file_dir="sam_mask_label",
):
    # Normalize & fix file-vs-folder input
    data_folder = os.path.normpath(data_folder)
    if os.path.isfile(data_folder):
        data_folder = os.path.splitext(data_folder)[0]

    # Step 1: write per-SAM-mask categories
    predict_sam_label(
        [data_folder],
        category_txt,
        masks_path_name,
        new_mask_label_file_name,
        pred_mask_file_name,
        sam_mask_label_file_dir,
    )

    # Colors
    color_list = np.load(color_list_path).astype(np.uint8)
    color_list[0] = [238, 239, 20]  # background

    # Walk subfolders
    for img_folder in os.listdir(data_folder):
        if img_folder == "sam_process.log":
            continue

        full_dir = os.path.join(data_folder, img_folder)
        if not os.path.isdir(full_dir):
            continue  # only sub-dirs

        category_info_path = os.path.join(
            full_dir, sam_mask_label_file_dir, new_mask_label_file_name
        )
        if not os.path.exists(category_info_path):
            logging.warning(f"Missing category info: {category_info_path} — skip.")
            continue

        # pred_mask path (with fallback)
        pred_mask_path = os.path.join(full_dir, pred_mask_file_name)
        if not os.path.exists(pred_mask_path):
            import glob

            candidates = glob.glob(
                os.path.join(full_dir, "**", pred_mask_file_name), recursive=True
            )
            if candidates:
                pred_mask_path = candidates[0]
            else:
                logging.warning(f"Missing pred mask: {pred_mask_path} — skip.")
                continue

        img_path = os.path.join(full_dir, "input.jpg")
        save_dir = full_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, enhance_mask_name)
        vis_save_path = os.path.join(save_dir, enhance_mask_vis_name)

        # Read & sort categories
        with open(category_info_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= 1:
            logging.warning(f"No category rows: {category_info_path} — skip.")
            continue

        category_info = lines[1:]
        category_area = np.zeros((num_class,), dtype=np.float32)
        for info in category_info:
            parts = info.strip().split(",")
            if len(parts) < 5:
                continue
            label = int(parts[1])
            area = float(parts[4])
            if 0 <= label < num_class:
                category_area[label] += area

        category_info = sorted(
            category_info, key=lambda x: float(x.strip().split(",")[4]), reverse=True
        )[:top_k]

        # Base mask
        pred_mask = _read_mask_channel_r(pred_mask_path)
        enhanced_mask = pred_mask.copy()

        # Load SAM masks
        masks_full_path = os.path.normpath(os.path.join(full_dir, masks_path_name))
        if not os.path.exists(masks_full_path):
            logging.warning(f"Missing masks.npy: {masks_full_path} — skip.")
            continue
        sam_masks = np.load(masks_full_path)

        # Apply selected SAM regions
        for info in category_info:
            parts = info.strip().split(",")
            if len(parts) < 5:
                continue
            idx = int(parts[0])
            label = int(parts[1])
            count_ratio = float(parts[3])
            area = float(parts[4])

            if area < area_thr or count_ratio < ratio_thr:
                continue

            sam_mask = sam_masks[idx].astype(bool)
            # Soft check area (avoid crash due to rounding)
            frac = sam_mask.sum() / (sam_mask.shape[0] * sam_mask.shape[1])
            if abs(frac - area) > 5e-3:
                logging.debug(
                    f"Area mismatch idx={idx}: sam={frac:.4f} vs info={area:.4f}"
                )

            enhanced_mask[sam_mask] = label

        # Save results
        cv2.imwrite(os.path.normpath(save_path), enhanced_mask)
        visualization_save(enhanced_mask, vis_save_path, img_path, color_list)
