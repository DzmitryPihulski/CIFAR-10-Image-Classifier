import random
from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms  # type: ignore


def set_seed(seed_value: int):
    """
    Set the seed for reproducibility. This function ensures that the results
    are consistent across different runs by setting the seed for the random number
    generators used by Python, NumPy, and PyTorch (including GPU if available).

    Arguments:
    - seed_value: The integer value to set as the seed for all random number generators.
    """

    random.seed(seed_value)
    torch.manual_seed(seed_value)  # type: ignore
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class_number_to_name = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def show(item: Any, ax: Any, title: str | None = None):
    """
    Display an image on the given axis (`ax`) with an optional title. If no title is
    provided, the class label is used.

    Arguments:
    - item: Tuple of image tensor and label.
    - ax: Matplotlib axis to display the image.
    - title: Optional title; defaults to class label.
    """

    img, label = item
    ax.imshow(np.transpose(img.numpy(), (1, 2, 0)), interpolation="nearest")
    if title:
        ax.set_title(label=title)
    else:
        ax.set_title(label=f"{class_number_to_name[label].capitalize()}")


transform_pipeline = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(degrees=15),
    ]
)
