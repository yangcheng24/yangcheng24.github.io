# Paper analysis on PointCLIP

In this blog post, I will review the paper Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos published in AAAI 2019 [1]. After briefly introducing the topic and relevant concepts, I will explain the method in my own words. Then we will discuss the results and future works.

![Image](https://example.com/my-coding-journey.jpg)

## Introduction

Deep learning has undoubtedly revolutionized various computer vision tasks across both 2D and 3D domains, tackling problems such as image classification, object detection, semantic segmentation, and point cloud recognition. However, the evolving world of 3D technology, particularly with the advent of sophisticated 3D sensing, is continually pushing the boundaries of what deep learning models can achieve. 

One specific challenge in the 3D world is dealing with point clouds - sets of data points in space that represent objects. Unlike 2D image data, 3D point clouds often suffer from space sparsity and irregular distribution, making it challenging to directly apply methods from the 2D realm. Even more interestingly, many newly captured point clouds contain objects from "unseen" categories, i.e., objects the model hasn't been trained on. This opens up a real challenge since even the best classifier might fail to recognize these objects, and re-training models each time when these "unseen" objects arise can be quite impractical.

In contrast, 2D vision tasks have made significant progress in mitigating similar issues, particularly through the use of Contrastive Vision-Language Pre-training (CLIP). By correlating vision and language, CLIP has shown promising results for zero-shot classification of "unseen" categories. Further enhancements have been achieved through the use of learnable tokens (CoOp), lightweight residual-style adapters (CLIP-Adapter), and efficiency improvements (Tip-Adapter). 

This naturally leads us to a question: can such promising methods be successfully transferred to the more challenging domain of 3D point clouds? In this blog post, we introduce PointCLIP, a novel model that addresses this question by transferring CLIP's 2D pre-trained knowledge to 3D point cloud understanding. 


## Method

### Terminology

- Zero-shot classification: the model recognize a certain class, without having been trained on any sample of this class before.
- Few-shot classification: the model recognize a certain class, with having been trained on only a few samples of this class before.

As I said in the Introduction, PointCLIP tries to transfer the pretrained knowledge in 2D image in CLIP to the recognition of 3D point clouds, so it is necessary to introduce the main principles of CLIP here.
### A revisit to CLIP

CLIP (Contrastive Language–Image Pretraining) is a multimodal vision-language model developed by OpenAI. Its goal is to understand the relationship between images and text, which is achieved by representing images and text in the same embedding space. In this space, associated images and texts are mapped closer together. Now, let me explain how it works by dividing its pipeline into three parts and explaining each part one by one.

- Contrastive Pre-Training: CLIP is trained with a contrastive learning approach. This technique pushes the model to identify which data are similar and which are different. CLIP is trained to bring related images and text closer while distancing unrelated images and text in the embedding space.
- Creating a Dataset Classifier from Label Text: After CLIP is pre-trained, we get the optimized text- and image encoders, then it can be used for downstream tasks, such as image classification, without task-specific fine-tuning. All possible classes the object on the input image may belong to are converted into textual prompts, and we input these textual prompts to the text-encoder, then we get the vectors of these textual prompts in the embedding space.
- Use for Zero-Shot Prediction: We input unseen image to the image-encoder, then we get the vector of this image in the embedding space, after that we multiply this vector with the vectors of those textual prompts one by one, the biggest one among these multiplications implicates the class the unseen image belongs to.

Now, we can finally introduce our protagonist, PointCLIP. The proposed PointCLIP is designed to address the disparity between the scale and diversity of 2D and 3D datasets, aiming to improve the understanding of 3D point cloud data. The primary idea behind PointCLIP is to leverage the pre-trained knowledge from the CLIP model, using it to carry out zero-shot learning on point clouds.

To facilitate this, point clouds are converted into representations that are compatible with CLIP by generating point-projected images from multiple views. This step bridges the modal gap between 2D and 3D data, making the point clouds easier for the model to process. Importantly, this is a cost-effective approach that doesn't require pre-transformation of the data, further enhancing its practical value.

### Main contributions

Let's have a look at the main contributions of the paper:

- Proposes PointCLIP to extend CLIP for handling 3D cloud data.
- An inter-view adapter is introduced upon PointCLIP and largely improves the performance by few-shot fine-tuning.
- PointCLIP can be utilized as a multi-knowledge ensemble module to enhance the performance of existing fully-trained 3D networks.

After listing the main contributions of the paper, let me explain zero-shot classification, few-shot classification and multi-knowledge ensembel, which conresponding to the first, second and last of the main contributions listed above, respectively.
