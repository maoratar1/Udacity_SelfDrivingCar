from ComputerVision.TheMachineLearningWorkflow.ChoosingMatrics.utils import get_data, check_results
from ComputerVision.TheMachineLearningWorkflow.ChoosingMatrics import iou
import numpy as np


def test_iou():
    a = np.array([1, 1, 4, 4])
    b = np.array([3, 3, 6, 6])
    assert iou.calculate_iou(a, b) == 1 / 17


def udacity_tests():
    ground_truth, predictions = get_data()
    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)

    ious = iou.calculate_ious(gt_bboxes, pred_boxes)
    check_results(ious)


if __name__ == "__main__":
    test_iou()
    print("Passed My tests")
    udacity_tests()