import random
import torch
import numpy as np

def set_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
class_number_to_name = {
    0:'airplane',										
1: "automobile",										
2: "bird",										
3:"cat",										
4: "deer",										
5: "dog",										
6: "frog",										
7: "horse",										
8: "ship",										
9: "truck"
}
        
def show(item, ax, title=None):
    img, label = item
    ax.imshow(np.transpose(img.numpy(), (1, 2, 0)), interpolation='nearest')
    if title:
        ax.set_title(label=title)
    else:
        ax.set_title(label=f"{class_number_to_name[label].capitalize()}")
        