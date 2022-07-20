from ComputerVision.TheMachineLearningWorkflow.ChoosingMatrics.iou import calculate_ious
from ComputerVision.TheMachineLearningWorkflow.ChoosingMatrics.precision_recall import precision_recall
from ComputerVision.TheMachineLearningWorkflow.ChoosingMatrics.utils import get_data

import numpy as np


def test_udacity():
    ground_truth, predictions = get_data()

    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    gt_classes = [g['classes'] for g in ground_truth if g['filename'] == filename][0]

    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    pred_classes = [p['classes'] for p in predictions if p['filename'] == filename][0]

    ious = calculate_ious(gt_bboxes, pred_boxes)
    precision, recall = precision_recall(ious, gt_classes, pred_classes)


if __name__ == "__main__":
    test_udacity()
