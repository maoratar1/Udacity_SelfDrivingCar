import numpy as np


def precision_recall(ious, gt_classes, pred_classes):
    """
    calculate precision and recall with using 0.5 as the iou threshold
    args:
    - ious [array]: NxM array of ious
    - gt_classes [array]: 1xN array of ground truth classes
    - pred_classes [array]: 1xM array of pred classes
    returns:
    - precision [float]
    - recall [float]
    """
    IOU_THERSHOLD = 0.5
    TP = 0
    FP = 0
    FN = 0

    xs, ys = np.where(ious > IOU_THERSHOLD)

    for x, y in zip(xs, ys):
        if pred_classes[x] == gt_classes[y]:
            TP += 1
        else:
            FP += 1

    FN = len(gt_classes) - len(np.unique(xs))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall
