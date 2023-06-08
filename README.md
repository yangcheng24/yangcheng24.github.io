# Paper analysis on PointCLIP

In this blog post, I will review the paper Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos published in AAAI 2019 [1]. After briefly introducing the topic and relevant concepts, I will explain the method in my own words. Then we will discuss the results and future works.

![Image](https://example.com/my-coding-journey.jpg)

## Introduction

Deep learning has undoubtedly revolutionized various computer vision tasks across both 2D and 3D domains, tackling problems such as image classification, object detection, semantic segmentation, and point cloud recognition. However, the evolving world of 3D technology, particularly with the advent of sophisticated 3D sensing, is continually pushing the boundaries of what deep learning models can achieve. 

One specific challenge in the 3D world is dealing with point clouds - sets of data points in space that represent objects. Unlike 2D image data, 3D point clouds often suffer from space sparsity and irregular distribution, making it challenging to directly apply methods from the 2D realm. Even more interestingly, many newly captured point clouds contain objects from "unseen" categories, i.e., objects the model hasn't been trained on. This opens up a real challenge since even the best classifier might fail to recognize these objects, and re-training models each time when these "unseen" objects arise can be quite impractical.

In contrast, 2D vision tasks have made significant progress in mitigating similar issues, particularly through the use of Contrastive Vision-Language Pre-training (CLIP). By correlating vision and language, CLIP has shown promising results for zero-shot classification of "unseen" categories. Further enhancements have been achieved through the use of learnable tokens (CoOp), lightweight residual-style adapters (CLIP-Adapter), and efficiency improvements (Tip-Adapter). 

This naturally leads us to a question: can such promising methods be successfully transferred to the more challenging domain of 3D point clouds? In this blog post, we introduce PointCLIP, a novel model that addresses this question by transferring CLIP's 2D pre-trained knowledge to 3D point cloud understanding. 

## 我的学习资源

我主要依赖以下资源来学习Python：

1. [Codecademy](https://www.codecademy.com/learn/learn-python-3)的Python 3课程
2. [Coursera](https://www.coursera.org/specializations/python)的Python专项课程
3. [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)书籍

## 我的第一个Python项目

我已经完成了我的第一个Python项目，它是一个简单的命令行游戏。以下是一段示例代码：

```python
def play_game():
    number_to_guess = 7
    guess = int(input("Guess a number between 1 and 10: "))
    
    if guess == number_to_guess:
        print("You guessed it!")
    else:
        print("Sorry, try again.")
