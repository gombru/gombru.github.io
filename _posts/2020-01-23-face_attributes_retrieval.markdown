---
layout: post
comments: true
title:  "Face Images Retrieval with Attributes Modifications"
excerpt: ""
date:   2020-01-023 20:00:00
img: "/assets/face_retrieval/intro.png"
mathjax: false
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

In this work I design an **image retrieval system able to retrieve images similar to a query image but with some modified attributes**. In our setup we employ a dataset of face images, and the attributes are annotated face characteristics, such as *smiling*, *young*, or *blond*. However, the proposed model and training could be used with other type of images and attributes.  
I design a model to learn embeddings of images and attributes to a joint space, test different training approaches, and evaluate if the system is able to retrieve images similar to a query image with an attribute modification, as shown in the following figure.

<div class="imgcap">
	<img src="/assets/face_retrieval/objective.png" width="460">
	<div class="thecap">
	On the left, query image with 2 different attributes modification. On the right, retrieval results.
	</div>
</div>

> The code used in this work is available [here](https://github.com/gombru/face_attributes_retrieval).


## Data
I use the [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset which contains arround **200k celebrity face images with 40 annotated binary attributes**. Standard dataset splits are employed: 162k train, 20k val and 20k test images.

<div class="imgcap">
	<img src="/assets/face_retrieval/dataset.png" width="460">
	<div class="thecap">
	CelebA dataset images sharing some attributes.
	</div>
</div>

 - **Images:** Use the already aligned and cropped images and resize them to 224x224 (to fit ResNet input size). Since they are aligned, I'll omit data augmentation.

 - **Attributes:** Encoded as 40-Dimensional multi-hot vectors.


## Model

The proposed model aims to learn embeddings for images and attributes vectors to a joint embedding space, where the embedding of a given image should be close to its attributes vector embedding. A pretrained ResNet-50, whose last layer has been replaced by a linear layer with 300 outputs, learns embeddings of images into out joint embedding space (which will have 300 dimensions). Attributes are processed by 2 linear layers with 300 outputs (the first one with Batch Norm and ReLu) and then embedded in the same space. Both the 300-D representations of images and attributes are L2-normalized.

<div class="imgcap">
	<img src="/assets/face_retrieval/model.png" width="460">
	<div class="thecap">
	Proposed face images retrieval with attributes modifications model.
	</div>
</div> 

In this figure, we use a siamese setup for attribute vectors $$a_x$$ and $$a_n$$, so the layers processing them are actually the same ($$\phi$$).

The reason of applying L2-normalization to the embeddings is that in test time, when we operate with query image embeddings and attribute representations, we won't need to care about the final query representation magnitude, since all the embeddings will be normalized. Normalizing embeddings reduces their discrimination capability, but given the high dimensionality of the embedding space that won't be a problem in our setup.

## Training

The model is trained with a [triplet margin loss](https://gombru.github.io/2019/04/03/ranking_loss/) (or triplet loss, ranking loss ...). If the anchor is an image $$I_x$$, the positive is its attribute vector $$a_x$$, and the negative is a different attribute vector $$a_n$$, the loss forces the distance between $$I_x$$ and $$a_n$$ in the embedding space to be larger than the distance between $$I_x$$ and $$a_x$$ by a margin. 

> For a detailed explanation of how triplet loss works, read the post ["Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss and all those confusing names"](https://gombru.github.io/2019/04/03/ranking_loss/).  

Intuitively, **the loss forces the image embedding to be closer in the embedding space to its attributes embedding than to other attribute embeddings**. Therefore, the embedding of an image of a blond man, will be closer in the embedding space to the *blond, man* attributes than to the *blond, woman* attributes or to the *black hair, man* attributes.  

I used a margin of 0.1 and an initial learning rate of 0.1, which is divided by 10 every 5 epochs.

### Negatives
I used two different negatives mining strategies:

 - **Random existing negatives:** Random (different) attribute vector in the dataset.

 - **Hard negatives:** To build the negative attributes $$a_n$$, change the value of 1 to 3 random attributes of the positive attributes vector $$a_x$$. Empirically, I found out that this hard negatives can only be used in one third of the triplets. If they are used more, the net overfits to real attribute vectors, embedding them always closer to the image because they are real, not because their attributes match. 


## Retrieval

In order to use the proposed model for retrieve images similar to a query image with some modified attributes, we follow this procedure: 

 - 1. Compute embeddings of the test images.

 - 2. Computing embedding of the query image.

 - 3. Compute embedding of the attribute vector which we want to use to modify our query image. As an example, if we want to modify the *black_hair* attribute and its index in our 40-dimensional attributes vector is $$0$$, this vector would be [1,0,0,0,0, ...]. 

 - 4. Combine the query image embedding and the attribute vector embedding. If we want to change the query image to have black hair, we would sum the vectors. If our query image has black hair and we want to retrieve similar ones but with other hair colors, we would subtract the attribute vector embedding (encoding *black_hair*) from the query image embedding.

 - 5. L2 normalize the resulting embedding, to ensure that, no matter the operations we have done, all the embeddings have the same magnitude.

 - 6. Compute the distance between the modified query embedding and each test image embedding, and get the most similar images. The distances used here is 
 the euclidean distance (and should be) the same as the one used in the loss (in this case torch' [TripletMarginLoss](https://pytorch.org/docs/stable/nn.html#tripletmarginloss)).

 > When combining query image embeddings and attribute vectors in step 4, we can do any operations we want, adding or subtracting different attributes, or even weighting image and attribute embeddings to increase their influence in the final embedding.


## Evaluation

This work stayed as a fast toy experiments, so I've not compare its performance or methodology with related work in the field. However, I designed performance metrics to **evaluate the trained models in both similarity of the retrieved images with the query image and with the desired target attributes** (those are the query image attributes with the modifications). First, I generate random queries by selecting a random attribute modification for each query image. Then, I retrieve the top-10 images for those queries, and compute the following evaluation metrics: 

 - **Mean Correct Attributes (MCA):** Mean number of results (we consider 10 per query) that have the specified attribute by the attribute modifier. This measure does not evaluate similarity with original image. 

 - **Mean Cosine Similarity (MCS):** Mean cosine similarity of the retrieved images (10) with the query images in the embedding space. This measure does not evaluate attribute modifications. 

 - **Cosine Similarity P@10 (CS-P@10):** A result is considered relevant if it has the attribute specified by the modifier. Then, it contributes to [P@10](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)) with its cosine similarity with the query image in the embedding space, to evaluate also the similarity with the query image. This measure evaluates both similarity with the query image and the attribute modification.


<div class="imgcap">
	<img src="/assets/face_retrieval/results.png" width="460">
	<div class="thecap">
	Results. Attribute multiplier, stands for a scalar multiplied by the attributes vector embedding in step 4., in order to control its influence in the resulting query embedding.
	</div>
</div> 

When we use an attribute multiplier of $$2$$ (which means that in step 4. the weight of the attributes vector is doubled), the MCA performance increases, since we are increasing the influence of the modified attribute in the resulting embedding. Similarly, if we decrease its influence multiplying it by $$0.5$$, we'll retrieve images more similar to the query image, and so the MCS performance increases. Results also show that the hard negatives boost performance.


## Qualitative results

Following, I show some qualitative results of the best performing model, where a query with a single attribute modification is done and the top-3 results are shown.

<div class="imgcap">
	<img src="/assets/face_retrieval/male.png" width="460">
</div> 

<div class="imgcap">
	<img src="/assets/face_retrieval/black_hair.png" width="460">
</div> 

<div class="imgcap">
	<img src="/assets/face_retrieval/eyeglasses.png" width="460">
</div> 

<div class="imgcap">
	<img src="/assets/face_retrieval/no_eyeglasses.png" width="460">
</div> 

<div class="imgcap">
	<img src="/assets/face_retrieval/no_male.png" width="460">
</div> 

<div class="imgcap">
	<img src="/assets/face_retrieval/no_young.png" width="460">
</div> 



