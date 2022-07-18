from ComputerVision.TheMachineLearningWorkflow.ChoosingMatrics import iou
import numpy as np


def test_iou():
    a = np.array([1, 1, 4, 4])
    b = np.array([3, 3, 6, 6])
    assert iou.calculate_iou(a, b) == 1 / 17


if __name__ == "__main__":
    test_iou()
    print("All tests passed")