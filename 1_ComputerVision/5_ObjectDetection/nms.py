import json

from utils import calculate_iou, check_results


def nms(predictions):
    """
    non max suppression
    args:
    - predictions [dict]: predictions dict 
    returns:
    - filtered [list]: filtered bboxes and scores
    """
    IOU_THERSHOLD = 0.5
    filtered = []
    boxes = predictions["boxes"]
    scores = predictions["scores"]

    for i in range(len(boxes)):
        discard = False
        for j in range(i + 1, len(boxes)):
            if calculate_iou(boxes[i], boxes[j]) > IOU_THERSHOLD:
                if scores[i] < scores[j]:
                    discard = True
                    break

        if not discard:
            filtered.append([boxes[i], scores[i]])

    return filtered


if __name__ == '__main__':
    with open('data/predictions_nms.json', 'r') as f:
        predictions = json.load(f)
    
    filtered = nms(predictions)
    check_results(filtered)