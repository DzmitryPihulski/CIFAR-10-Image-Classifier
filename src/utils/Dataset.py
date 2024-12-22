import pickle
from typing import Any

import torch
import torch.utils.data as data
import torchvision.transforms as transforms  # type: ignore


def unpickle(batch_path: str) -> dict[Any, Any] | str:
    """
    Unpickle a batch of images in bin format from a file.
    :param batch_path:
    :return:
    """
    try:
        with open(batch_path, "rb") as fo:
            file: dict[Any, Any] = pickle.load(fo, encoding="bytes")
    except FileNotFoundError:
        return "There is no such file or directory"
    return file


class CustomImageDataset(data.Dataset):  # type: ignore
    """
    Custom dataset for image classification.
    """

    def __init__(self):
        """
        Image dataset constructor.
        """
        self.data: dict[Any, Any] = dict()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return list(self.data.keys())[idx], self.data[list(self.data.keys())[idx]]

    def load_batch(self, batch_path: str) -> str | None:
        """
        Load a batch of images from a file.
        :param batch_path:
        :return: None
        """
        batch = unpickle(batch_path)
        try:
            for i in range(len(batch[b"labels"])):  # type: ignore
                red_channel = torch.Tensor(batch[b"data"][i][:1024]).view(32, 32)  # type: ignore
                green_channel = torch.Tensor(batch[b"data"][i][1024:2048]).view(32, 32)  # type: ignore
                blue_channel = torch.Tensor(batch[b"data"][i][2048:]).view(32, 32)  # type: ignore
                self.data[
                    torch.stack((red_channel, green_channel, blue_channel)) / 255
                ] = batch[b"labels"][i]  # type: ignore
        except KeyError:
            return "KeyError: There is not such file or directory"
        return None

    def load_batch_transformed(
        self, batch_path: str, transform: transforms.Compose
    ) -> str | None:
        """
        Load a batch of images and transfrom from a file.
        :param batch_path:
        :return: None
        """
        batch = unpickle(batch_path)
        try:
            for i in range(len(batch[b"labels"])):  # type: ignore
                red_channel = torch.Tensor(batch[b"data"][i][:1024]).view(32, 32)  # type: ignore
                green_channel = torch.Tensor(batch[b"data"][i][1024:2048]).view(32, 32)  # type: ignore
                blue_channel = torch.Tensor(batch[b"data"][i][2048:]).view(32, 32)  # type: ignore
                self.data[
                    transform(
                        torch.stack((red_channel, green_channel, blue_channel)) / 255
                    )
                ] = batch[b"labels"][i]  # type: ignore
        except KeyError:
            return "KeyError: There is not such file or directory"
        return None

    def get_class_images(self, class_label: int) -> list[Any]:
        """
        Get all images of a specific class.
        :param class_label: The class label to retrieve images for
        :return: List of images (as tensors) belonging to the specified class
        """
        return [
            [image, label] for image, label in self.data.items() if label == class_label
        ]
