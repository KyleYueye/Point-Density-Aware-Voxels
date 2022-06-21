import io as sysio

import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval


class EvaluationLevels:
    funcs = {
        'norm': lambda x: np.linalg.norm(x, axis=1)
    }

    def __init__(self, eval_levels_cfg):
        self.annotation = eval_levels_cfg.ANNOTATION
        self.num_levels = eval_levels_cfg.NUM_LEVELS
        self.levels = eval_levels_cfg.LEVELS
        self.func = eval_levels_cfg.get('FUNC', None)
        self.filter = eval_levels_cfg.get('FILTER', None)
        self.ignore_kitti_filter = eval_levels_cfg.get('IGNORE_KITTI_FILTER', False)

        for class_name, level in self.levels.items():
            assert self.num_levels+1 == len(level)
        if self.func:
            assert self.func in EvaluationLevels.funcs
            self.func = EvaluationLevels.funcs[self.func]
        if self.filter:
            self.filter = {
                'annotation': self.filter.ANNOTATION,
                'value': self.filter.VALUE
            }

    def check_filter(self, anno, index):
        if not self.filter:
            return True

        assert self.filter['annotation'] in anno
        if anno[self.filter['annotation']][index] == self.filter['value']:
            return True
        return False

    def ignore_anno(self, anno, index, current_cls_name, difficulty):
        assert self.annotation in anno

        if not self.check_filter(anno, index):
            return True

        current_cls_name = current_cls_name.upper()
        levels = self.levels['DEFAULT'] if current_cls_name not in self.levels else self.levels[current_cls_name]

        anno_val = anno[self.annotation] if self.func is None else self.func(anno[self.annotation])
        if anno_val[index] >= levels[difficulty] and anno_val[index] < levels[difficulty+1]:
            return False
        return True

    def ignore_gt(self, gt_anno, index, current_cls_name, difficulty):
        ignore = self.ignore_anno(gt_anno, index, current_cls_name, difficulty)
        if not self.ignore_kitti_filter:
            ignore |= ignore_gt_kitti(gt_anno, index, difficulty=-1)
        return ignore

    def ignore_dt(self, dt_anno, index, current_cls_name, difficulty):
        ignore = self.ignore_anno(dt_anno, index, current_cls_name, difficulty)
        if not self.ignore_kitti_filter:
            ignore |= ignore_dt_kitti(dt_anno, index, difficulty=-1)
        return ignore


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def ignore_gt_kitti(gt_anno, index, difficulty):
    """
    Indicates whether to ignore GT sample
    Args:
        gt_anno [dict]: Ground truth annotation
        index [int]: GT sample index
        difficulty [int]: Difficulty index
    """
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]

    # Compute box height
    bbox = gt_anno['bbox'][index]
    height = bbox[3] - bbox[1]

    ignore = False
    if ((gt_anno["occluded"][index] > MAX_OCCLUSION[difficulty])
            or (gt_anno["truncated"][index] > MAX_TRUNCATION[difficulty])
            or (height <= MIN_HEIGHT[difficulty])):
        ignore = True

    return ignore


def ignore_dt_kitti(dt_anno, index, difficulty):
    MIN_HEIGHT = [40, 25, 25]
    height = abs(dt_anno["bbox"][index, 3] - dt_anno["bbox"][index, 1])

    ignore = False
    if height < MIN_HEIGHT[difficulty]:
        ignore = True

    return ignore


def clean_data(gt_anno, dt_anno, current_class, difficulty, eval_levels=None):
    CLASS_NAMES = ['bus', 'car', 'bicycle', 'pedestrian', 'tricycle', 'semitrailer', 'truck']
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        # elif (current_cls_name == "pedestrian".lower()
        #       and "pedestrian".lower() == gt_name):
        #     valid_class = 0
        # elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
        #     valid_class = 0
        else:
            valid_class = -1
        # ignore = ignore_gt_kitti(gt_anno, i, difficulty) if eval_levels is None else eval_levels.ignore_gt(gt_anno, i, current_cls_name, difficulty)
        ignore = False
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        # ignore = ignore_dt_kitti(dt_anno, i, difficulty) if eval_levels is None else eval_levels.ignore_dt(dt_anno, i, current_cls_name, difficulty)
        if ignore:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty, eval_levels=None):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty, eval_levels)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=100,
               eval_levels=None):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty, eval_levels=eval_levels)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
    }
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            PR_detail_dict=None,
            eval_levels=None):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2] if eval_levels is None else [i for i in range(eval_levels.num_levels)]
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps, eval_levels=eval_levels)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']
    return mAP_3d, mAP_3d_R40


def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges,
                       compute_aos):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            # np.linspace must have an integer as 3rd argument
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:2, i, j], int(overlap_ranges[2, i, j]))
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap]
    mAPbbox = mAPbbox.mean(-1)
    mAPbbox_R40 = mAPbbox_R40.mean(-1)

    mAPbev = mAPbev.mean(-1)
    mAPbev_R40 = mAPbev_R40.mean(-1)

    mAP3d = mAP3d.mean(-1)
    mAP3d_R40 = mAP3d_R40.mean(-1)

    if mAPaos is not None:
        mAPaos = mAPaos.mean(-1)
        mAPaos_R40 = mAPaos_R40.mean(-1)

    return mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40


def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None, eval_levels_list_cfg=None):
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: 'bus',
        1: 'car',
        2: 'bicycle',
        3: 'pedestrian',
        4: 'tricycle',
        5: 'semitrailer',
        6: 'truck'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break

    # Check if eval level is necessary
    eval_levelss = []
    if eval_levels_list_cfg:
        for eval_levels_cfg in eval_levels_list_cfg:
            eval_levelss.append(EvaluationLevels(eval_levels_cfg))

    ret_dict = {}
    if not eval_levelss:
        mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
            gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)
        for j, curcls in enumerate(current_classes):
            # mAP threshold array: [num_minoverlap, metric, class]
            # mAP result: [num_class, num_diff, num_minoverlap]
            for i in range(min_overlaps.shape[0]):
                result += print_str(
                    (f"{class_to_name[curcls]} "
                    "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
                result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
                                    f"{mAPbbox[j, 1, i]:.4f}, "
                                    f"{mAPbbox[j, 2, i]:.4f}"))
                result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
                                    f"{mAPbev[j, 1, i]:.4f}, "
                                    f"{mAPbev[j, 2, i]:.4f}"))
                result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
                                    f"{mAP3d[j, 1, i]:.4f}, "
                                    f"{mAP3d[j, 2, i]:.4f}"))

                if compute_aos:
                    result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
                                        f"{mAPaos[j, 1, i]:.2f}, "
                                        f"{mAPaos[j, 2, i]:.2f}"))
                    # if i == 0:
                    # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
                    # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
                    # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]

                result += print_str(
                    (f"{class_to_name[curcls]} "
                    "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
                result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
                                    f"{mAPbbox_R40[j, 1, i]:.4f}, "
                                    f"{mAPbbox_R40[j, 2, i]:.4f}"))
                result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
                                    f"{mAPbev_R40[j, 1, i]:.4f}, "
                                    f"{mAPbev_R40[j, 2, i]:.4f}"))
                result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
                                    f"{mAP3d_R40[j, 1, i]:.4f}, "
                                    f"{mAP3d_R40[j, 2, i]:.4f}"))
                if compute_aos:
                    result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
                                        f"{mAPaos_R40[j, 1, i]:.2f}, "
                                        f"{mAPaos_R40[j, 2, i]:.2f}"))
                    if i == 0:
                        ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
                        ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
                        ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]

                if i == 0:
                    # ret_dict['%s_3d/easy' % class_to_name[curcls]] = mAP3d[j, 0, 0]
                    # ret_dict['%s_3d/moderate' % class_to_name[curcls]] = mAP3d[j, 1, 0]
                    # ret_dict['%s_3d/hard' % class_to_name[curcls]] = mAP3d[j, 2, 0]
                    # ret_dict['%s_bev/easy' % class_to_name[curcls]] = mAPbev[j, 0, 0]
                    # ret_dict['%s_bev/moderate' % class_to_name[curcls]] = mAPbev[j, 1, 0]
                    # ret_dict['%s_bev/hard' % class_to_name[curcls]] = mAPbev[j, 2, 0]
                    # ret_dict['%s_image/easy' % class_to_name[curcls]] = mAPbbox[j, 0, 0]
                    # ret_dict['%s_image/moderate' % class_to_name[curcls]] = mAPbbox[j, 1, 0]
                    # ret_dict['%s_image/hard' % class_to_name[curcls]] = mAPbbox[j, 2, 0]

                    ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
                    ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
                    ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
                    ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
                    ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
                    ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
                    ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
                    ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
                    ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]
    else:
        for eval_levels in eval_levelss:
            mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
                gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict, eval_levels=eval_levels)

            result_dict = {}
            for j, curcls in enumerate(current_classes):
                class_name = class_to_name[curcls]
                class_dict = {}
                class_dict['levels'] = eval_levels.levels.DEFAULT if class_name.upper() not in eval_levels.levels else eval_levels.levels[class_name.upper()]
                class_dict['3d'] = mAP3d_R40[j, :, 0]
                class_dict['bev'] = mAPbev_R40[j, :, 0]
                class_dict['image'] = mAPbbox_R40[j, :, 0]
                if compute_aos:
                    class_dict['aos'] = mAPaos_R40[j, :, 0]

                result_dict[class_name] = class_dict
            level_annotation = eval_levels.annotation
            level_annotation += "_{}_{}".format(eval_levels.filter['annotation'], eval_levels.filter['value']) if eval_levels.filter else ""
            ret_dict[level_annotation] = result_dict

    return result, ret_dict


def get_coco_eval_result(gt_annos, dt_annos, current_classes):
    class_to_name = {
        0: 'bus',
        1: 'car',
        2: 'bicycle',
        3: 'pedestrian',
        4: 'tricycle'
    }
    
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break

    ret_dict = {}

    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_coco_style_eval(
        gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)

        # Print R11
        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox[j, 0]:.2f}, "
                             f"{mAPbbox[j, 1]:.2f}, "
                             f"{mAPbbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{mAPaos[j, 0]:.2f}, "
                                 f"{mAPaos[j, 1]:.2f}, "
                                 f"{mAPaos[j, 2]:.2f}"))

        # Print R40
        result += print_str((f"{class_to_name[curcls]} "
                             "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox_R40[j, 0]:.4f}, "
                            f"{mAPbbox_R40[j, 1]:.4f}, "
                            f"{mAPbbox_R40[j, 2]:.4f}"))
        result += print_str((f"bev  AP:{mAPbev_R40[j, 0]:.4f}, "
                            f"{mAPbev_R40[j, 1]:.4f}, "
                            f"{mAPbev_R40[j, 2]:.4f}"))
        result += print_str((f"3d   AP:{mAP3d_R40[j, 0]:.4f}, "
                            f"{mAP3d_R40[j, 1]:.4f}, "
                            f"{mAP3d_R40[j, 2]:.4f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{mAPaos_R40[j, 0]:.2f}, "
                                f"{mAPaos_R40[j, 1]:.2f}, "
                                f"{mAPaos_R40[j, 2]:.2f}"))

        ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0]
        ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1]
        ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2]
        ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0]
        ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1]
        ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2]
        ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0]
        ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1]
        ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2]

        if compute_aos:
            ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0]
            ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1]
            ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2]

    return result, ret_dict


def get_wanji_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None, eval_levels_list_cfg=None):
    overlap_0_5 = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    overlap_0_25 = np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                             [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                             [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
    min_overlaps = np.stack([overlap_0_5, overlap_0_25], axis=0)  # [2, 3, 7]
    class_to_name = {
        0: 'bus',
        1: 'car',
        2: 'bicycle',
        3: 'pedestrian',
        4: 'tricycle',
        5: 'semitrailer',
        6: 'truck'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    # for anno in dt_annos:
    #     if anno['alpha'].shape[0] != 0:
    #         if anno['alpha'][0] != -10:
    #             compute_aos = True
    #         break

    # Check if eval level is necessary
    eval_levelss = []
    if eval_levels_list_cfg:
        for eval_levels_cfg in eval_levels_list_cfg:
            eval_levelss.append(EvaluationLevels(eval_levels_cfg))

    ret_dict = {}
    if not eval_levelss:
        mAP3d, mAP3d_R40 = do_eval(
            gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)
        for j, curcls in enumerate(current_classes):
            # mAP threshold array: [num_minoverlap, metric, class]
            # mAP result: [num_class, num_diff, num_minoverlap]
            for i in range(min_overlaps.shape[0]):
                result += print_str(
                    (f"\n{class_to_name[curcls]} " f"AP@{min_overlaps[i, 0, j]:.2f}, AP_R40@{min_overlaps[i, 0, j]:.2f}:"))
                result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f},     {mAP3d_R40[j, 0, i]:.4f}"))

                # result += print_str(
                #     (f"\n{class_to_name[curcls]} "
                #     "AP_R40@{:.2f}:".format(*min_overlaps[i, 0, j])))
                # result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "))
                
    else:
        for eval_levels in eval_levelss:
            mAP3d, mAP3d_R40 = do_eval(
                gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict, eval_levels=eval_levels)

            result_dict = {}
            for j, curcls in enumerate(current_classes):
                class_name = class_to_name[curcls]
                class_dict = {}
                class_dict['levels'] = eval_levels.levels.DEFAULT if class_name.upper() not in eval_levels.levels else eval_levels.levels[class_name.upper()]
                class_dict['3d'] = mAP3d_R40[j, :, 0]
                if compute_aos:
                    class_dict['aos'] = mAPaos_R40[j, :, 0]

                result_dict[class_name] = class_dict
            level_annotation = eval_levels.annotation
            level_annotation += "_{}_{}".format(eval_levels.filter['annotation'], eval_levels.filter['value']) if eval_levels.filter else ""
            ret_dict[level_annotation] = result_dict

    return result, ret_dict