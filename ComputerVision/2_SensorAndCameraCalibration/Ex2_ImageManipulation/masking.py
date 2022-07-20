import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    img = np.asarray(Image.open(path).convert('RGB'))
    img_R_ch, img_G_ch, img_B_ch = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = (img_R_ch > color_threshold[0]) & (img_G_ch > color_threshold[1]) & (img_B_ch > color_threshold[2])
    return img, mask


def mask_and_display(img, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """
    masked_image = np.copy(img)
    masked_image[:, :, 0][mask is False] = 0
    masked_image[:, :, 1][mask is False] = 0
    masked_image[:, :, 2][mask is False] = 0

    f, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[2].imshow(masked_image)

    plt.show()


if __name__ == '__main__':
    path = 'data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)