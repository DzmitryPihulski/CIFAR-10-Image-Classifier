# The CIFAR-10 dataset classification

## Description

In this repo, some basic models of deep learning were compared in the classification:

[The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

The models are:

| Model                                    | Number of Parameters | Best accuracy (%) |
| ---------------------------------------- | -------------------- | ----------------- |
| **Simple CNN model**                     | 62k                  | 62                |
| **More complex CNN model with drop-out** | 1.1M                 | 79                |
| **ResNet18**                             | 11.1M                | 85                |
| **VGG11**                                | 28.4M                | 86                |

Models architecture was coded in [models file](/utils/models.py).

The training dataset was augmented with [transform_pipeline](/utils/utils.py)

Examples of transfromations are:

<img src="src/data/images/transform_frog.png" width="700" />
<img src="src/data/images/transform_truck.png" width="700" />

<br>
You can read about the project and it's results in:

[main file(main.ipynb)](/main.ipynb)

## Set up and configuration

You can download the repository via the command

```
git clone https://github.com/DzmitryPihulski/LLM_question_and_answer_system_with_RAG.git
```

I used python version 3.11.4.
