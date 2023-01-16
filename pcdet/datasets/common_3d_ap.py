from multiprocessing import Pool

import numpy as np
from terminaltables import AsciiTable
from ..ops.iou3d_nms import iou3d_nms_utils
import torch


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).
    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]
    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 iou_thr=0.5,
                 area_ranges=None,
                 bev_mode=None,
                 use_legacy_coordinate=False):
    extra_length = 0.

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    if gt_bboxes.shape[0] == 0:
        fp[...] = 1
        return tp, fp

    torch_det_bboxes = torch.from_numpy(det_bboxes[:, :-1]).float().cuda()
    if torch_det_bboxes.size(0) == 0:
        torch_det_bboxes = torch.zeros((1, 7)).float().cuda()
    torch_gt_bboxes = torch.from_numpy(gt_bboxes).float().cuda()

    if bev_mode:
        ious = iou3d_nms_utils.boxes_iou_bev(torch_det_bboxes, torch_gt_bboxes)
    else:
        ious = iou3d_nms_utils.boxes_iou3d_gpu(torch_det_bboxes, torch_gt_bboxes)

    ious = ious.cpu().numpy()

    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):  # (None, None)
        gt_covered = np.zeros(num_gts, dtype=bool)

        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[k, i] = 1
                else:
                    fp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
    return tp, fp


def get_cls_results(det_results, annotations, class_id):
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = [img_res[class_id] for img_res in annotations]
    return cls_dets, cls_gts


def eval_map(det_results,
             annotations,
             scale_ranges=None,
             iou_thr=0.5,
             dataset='voc07',
             logger=None,
             tpfp_fn=tpfp_default,
             nproc=4,
             bev_mode=None,
             use_legacy_coordinate=False):
    """Evaluate mAP of a dataset.
    https://github.com/open-mmlab/mmdetection/blob/c88509cb9a73d6bd1edcba64eb924d3cf3cfe85d/mmdet/core/evaluation/mean_ap.py#L297
    """
    assert len(det_results) == len(annotations)
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.

    num_imgs = len(det_results)
    num_scales = 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts = get_cls_results(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tp, fp = [], []
        for single_det, single_gt in zip(cls_dets, cls_gts):
            single_tp, single_fp = tpfp_fn(single_det, single_gt, iou_thr, area_ranges, bev_mode, use_legacy_coordinate)
            tp.append(single_tp)
            fp.append(single_fp)

        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            num_gts[0] += bbox.shape[0]

        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })

    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if aps else 0.0

    return mean_ap, eval_results


def eval_bev_map(det_results,
                 annotations,
                 scale_ranges=None,
                 iou_thr=0.5,
                 dataset='voc07',
                 logger=None,
                 tpfp_fn=tpfp_default,
                 nproc=4,
                 bev_mode=True,
                 use_legacy_coordinate=False):
    """Evaluate mAP of a dataset.
    https://github.com/open-mmlab/mmdetection/blob/c88509cb9a73d6bd1edcba64eb924d3cf3cfe85d/mmdet/core/evaluation/mean_ap.py#L297
    """
    assert len(det_results) == len(annotations)
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.

    num_imgs = len(det_results)
    num_scales = 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts = get_cls_results(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tp, fp = [], []
        for single_det, single_gt in zip(cls_dets, cls_gts):
            single_tp, single_fp = tpfp_fn(single_det, single_gt, iou_thr, area_ranges, bev_mode, use_legacy_coordinate)
            tp.append(single_tp)
            fp.append(single_fp)

        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            num_gts[0] += bbox.shape[0]

        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })

    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if aps else 0.0

    return mean_ap, eval_results
