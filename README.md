# Self-Driving Cars - Udacity

# Chapter 1 - Computer Vision
## The Machine Learning Workflow 
### Ex1 | Choosing metrics

The bounding boxes are using the `[x1, y1, x2, y2]` format. 

#### Part 1 - Calculate IOU

In the first part of this exercise, your task is to implement a function that calculates the iou between
two bounding boxes.

The `calculate_ious` function in `iou.py` takes two arrays containing the bounding boxes coordinates
as inputs. Both arrays are 1x4 numpy arrays. The array are using the following format:
```
[x1, y1, x2, y2]
```
where `x1 < x2` and `y1 < y2`. `(x, y1)` are the coordinates of the upper left corner 
and `(x2, y2)` the coordinates of the lower right corner of the bounding box.


#### Part 2 - calculate Precision / Recall

Then, you are asked to calculate the precision and recall for a given set of predictions 
and ground truths. You will use a threshold of 0.5 IoU to determine if a prediction is 
a true positive or not.

The `precision_recall` function in `precision_recall.py` takes as inputs a `ious` NxM array of IoU values as well as 
two list `pred_classes` and`gt_classes` containing the M predicted classes ids and the N ground truth classes ids.

The `ious` array contains the pairwise IoU values between the N ground truth bounding boxes and the M 
predicted bounding boxes.

### Exercise 2 - Visualization

For this exercise, you need to implement a function to visualize the ground truth boxes
on a set of images in `visualization.py`. You need to display color coded bounding boxes using the class id associated
with each bounding box. You need to display all the data in a single figure.
You should aim for visibility as clear data visualization is critical to communicate a message.

![](ComputerVision/TheMachineLearningWorkflow/DataAcquistionAndVisualization/example.png 
)


The labels (bounding boxes and classes) are located is the `data/ground_truth.json` file. It contains 20 observations, each observation is a dict
with the following fields.

```
{filename: str, boxes: List[List[int]], classes: List[int]}
```
The bounding boxes are using the `[x1, y1, x2, y2]` format. Images (png files) are located in the `data/images` folder. Each image is associated can be matched with its labels with the filename. 

The `utils.py` file contains an help function `get_data` that you can import to load the ground truth and the predictions. You will only need the ground truth for 
this exercise though. 

