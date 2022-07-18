from ComputerVision.TheMachineLearningWorkflow.DataAcquistionAndVisualization.utils import get_data
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    keys: "filename", "boxes", "classes"
    """
    rows = 4
    cols = 5
    f, ax = plt.subplots(rows, cols, figsize=(20, 10))  # len(ground_truth) == 20

    i, j = 0, 0

    classes_colors = {1: "red", 2: "green"}

    for gt_ind, gt in enumerate(ground_truth):
        i, j = gt_ind % rows, gt_ind % cols

        image_path = "data/images/" + gt["filename"]
        image = Image.open(image_path)

        ax[i, j].imshow(image)

        for cls, box in zip(gt["classes"], gt["boxes"]):
            y1, x1, y2, x2 = box

            rectangle = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  facecolor='none',
                                  edgecolor=classes_colors[cls])
            ax[i, j].add_patch(rectangle)

        ax[i, j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ground_truth, _ = get_data()
    viz(ground_truth)
