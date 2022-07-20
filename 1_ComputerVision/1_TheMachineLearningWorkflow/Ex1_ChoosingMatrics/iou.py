import numpy as np


def calculate_ious(gt_bboxes, pred_bboxes):
    """
    calculate ious between 2 sets of bboxes 
    args:
    - gt_bboxes [array]: Nx4 ground truth array
    - pred_bboxes [array]: Mx4 pred array
    returns:
    - iou [array]: NxM array of ious
    """
    ious = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i,j] = calculate_iou(gt_bbox, pred_bbox)
    return ious


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """
    x1, y1, x2, y2 = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
    x3, y3, x4, y4 = pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]
    
    x1_intrsct, x2_intrsct = max(x1, x3), min(x2, x4)
    intrsct_width = max(0, x2_intrsct - x1_intrsct)
    
    y1_intrsct, y2_intrsct = max(y1, y3), min(y2, y4)
    intrsct_height = max(0, y2_intrsct - y1_intrsct)
    
    intrsct_space = intrsct_height * intrsct_width
    
    pred_bbox_space = (x2 - x1) * (y2 - y1)
    gt_bbox_space = (x4 - x3) * (y4 - y3)
    
    union_space = pred_bbox_space + gt_bbox_space - intrsct_space
    
    iou = intrsct_space / union_space 
    
    return iou