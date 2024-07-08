import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def show_image(image_tensor):
    plt.imshow(image_tensor.permute(1, 2, 0))
    return plt.show()
def show(img):
    img = img/255
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    new = unpickle('data/cifar-10-batches-py/data_batch_1')
    image = 0
    print(new[b'labels'][image])
    print(len(new[b'data'][image]))
    for i in range(len(new[b'labels'])):
        if new[b'labels'][i] == 5:
            red_chanel_of_first_image = torch.Tensor(new[b'data'][i][:1024]).view(32, 32)
            green_chanel_of_first_image = torch.Tensor(new[b'data'][i][1024:2048]).view(32, 32)
            blue_chanel_of_first_image = torch.Tensor(new[b'data'][i][2048:]).view(32, 32)

            image_tensor = torch.stack((red_chanel_of_first_image, green_chanel_of_first_image, blue_chanel_of_first_image))
            show(image_tensor)
            break