import matplotlib.pyplot as plt
from PIL import Image, ImageStat
import numpy as np
import glob
from utils import check_results
# import seaborn as sns


def calculate_mean_std(image_list):
    """
    Calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    stds = []
    means = []

    for image_path in image_list:
        image = Image.open(image_path).convert("RGB")
        img_stat = ImageStat.Stat(image)
        means.append(np.array(img_stat.mean))
        stds.append(np.array(img_stat.var) ** 0.5)

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    R_vals, G_vals, B_vals = [], [], []

    for img_path in image_list:
        img = np.asarray(Image.open(img_path).convert("RGB"))
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        R_vals += list(R.flatten())
        G_vals += list(G.flatten())
        B_vals += list(B.flatten())

    sns.kdeplot(R_vals, color='r')
    sns.kdeplot(G_vals, color='g')
    sns.kdeplot(B_vals, color='b')
    plt.legend()
    plt.show()


if __name__ == "__main__": 
    image_list = glob.glob('data/images/*')
    mean, std = calculate_mean_std(image_list)
    channel_histogram(image_list[:2])
    check_results(mean, std)
