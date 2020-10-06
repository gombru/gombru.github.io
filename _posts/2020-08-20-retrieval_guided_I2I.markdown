---
layout: post
comments: true
title: "Retrieval Guided Unsupervised Multi-Domain Image to Image Translation"
excerpt: "We propose using an image retrieval system to boost the performance of an image to image translation system, experimenting with a dataset of face images."
date: 2020-08-20 20:00:00
img: "/assets/retrieval_guided_I2I.jpeg"
mathjax: false
---

<script type="text/javascript" async
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<span style="color:brown">**This work has been published in ACM MM 2020. Refer to the [paper](https://arxiv.org/abs/2008.04991) to access the full and formal article. Here I explain informally and briefly the experiments conducted and the conclusions obtained.**</span>

> The code used in this work is available [here](https://github.com/yhlleo/RG-UNIT).

This work builds over [GMM-UNIT](https://arxiv.org/abs/2003.06788), a model for Multi-Domain Image to Image Translation. GMM-UNIT learns to generate images in different domains learning from unpaired data. In order words: It learns to **generate an image similar to a given one but with different attributes**. When learning from face images with annotated attributes (i.e. man, blond, young), it learns to generate a face image similar to an input one but with different attributes (i.e. turn this young man old; turn this woman blond).

In this paper **we propose a method that improves GMM-UNIT performance by exploiting an image retrieval system that provides during training real images similar to the input one but with the desired target attributes**. The hypothesis is that GMM-UNIT can benefit from those real retrieved images to generate more realistic images.

<div class="imgcap">
<img src="/assets/retrieval_guided_I2I/I2I_results_comparison.png" width="600">
<div class="thecap">
Results of the original GMM-UNIT model and its proposed improvement (RG-UNIT).</div>
</div>

## Data
We use the [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset which contains around **200k celebrity face images with 40 annotated binary attributes**.

<div class="imgcap">
<img src="/assets/face_retrieval/dataset.png" width="600">
<div class="thecap">
CelebA dataset images sharing some attributes.
</div>
</div>

But only the following **attributes** are used: : **hair color (black/blond/brown), gender (male/female), and age (young/old)**, which constitute 7 domains.

## Procedure

We follow these steps:

1. **Train a standard GMM-UNIT I2I translation model**. This model implicitly **learns to extract attribute features** (encoding the image domain; i.e. man, blond, old) and **content features** (encoding the image content, which is the visual appearance not defined by the used attributes).

2. **Train an image retrieval system able to find images with the given content and attribute features** (using the feature exractors learned in step 1).

3. **Train the I2I translation model with the retrieval guidance (RG-UNIT). For a given training image and target attributes, the most similar images are found by the retrieval images, and exploited for the image generation**. (Importantly, the content and feature extractors are frozen in this step, to avoid damaging the retrieval performance).

Since step 1 is the standard training explained in [GMM-UNIT](https://arxiv.org/abs/2003.06788), we skip its explanation in this blog post. Just remember that GMM-UNIT learns a content features extractor (**$$E_c$$**) and an attributes features extractors (**$$E_a$$**) which are frocen from now on, and used in the following steps.

## Image Retrieval System Training

The image retrieval system is **trained with a [triplet loss](https://gombru.github.io/2019/04/03/ranking_loss/) to embed images with similar content features and attribute features nearby** in a vectorial space. You can learn about triplet loss and ranking losses in general [here](https://gombru.github.io/2019/04/03/ranking_loss/).

The retrieval system input is a concatenation of the content features and the attribute features.
To create the triplets to train our model, we exploit the GMM-UNIT I2I translation system trained in step 1, as explained in the following figure:

<div class="imgcap">
<img src="/assets/retrieval_guided_I2I/triplets.png" width="800">
</div>

- **Anchor**: The content features of a given image and the desired target attributes.
- **Positive**: The anchor image is translated to the desired attributes. The positive sample is formed by the content features and the attributes features of the generated image.

----

- **Easy Negative**: A random real image.
- **Medium Negative**: A generated image with the target anchor attributes given a random image as input.
- **Hard Negative**: A generated image with random attributes given the anchor image as input.
- **Hardest Negative**: A generated image with the anchor image original attributes given the anchor image as input.

The **negatives mining strategy is crucial for the training**, as shown in these results:

<div class="imgcap">
<img src="/assets/retrieval_guided_I2I/retrieval_quantitative_results.png" width="400">
</div>

Here some results of the trained retrieval system:

<div class="imgcap">
<img src="/assets/retrieval_guided_I2I/retrieval_results.png" width="750">
</div>

## Retrieval Guided I2I Translation Training

Now we have a retrieval system able to find real images with the desired content and attributes. During the I2I translation system training, we aim to **teach a model to generate images similar to an input one in content but with the desired attributes**. The retrieval guidance **provides the generator with real images similar to the input one in content but with the desired attributes**, and we aim that those images are useful to teach the generator to create more realistic images.

This is the training pipeline of the proposed retrieval guided image to image model:

<div class="imgcap">
<img src="/assets/retrieval_guided_I2I/full_pipeline.png" width="800">
</div>

The idea is simple: The retrieval system returns the images it founds and their content features are concatenated with the generator input. Importantly, the content and attributes feature extractors (**$$E_c$$, $$E_a$$**) are frozen during this training. Else, the retrieval system would fail.

## Quantitative Results

Quantitative results show that the retrieval guidance boosts GMM-UNIT performance in different image quality metrics:

<div class="imgcap">
<img src="/assets/retrieval_guided_I2I/I2I_quantitative_results.png" width="400">
</div>

An important and nice feature of the retrieval guidance, is that its **retrieval set is not limited to annotated images**. Therefore, the **retrieval guidance can benefit from additional unnanotated data**, which can possibly boost even more the performance. To simulate that scenario, we train the RG-UNIT with a subset of the dataset, but include the whole dataset as the retrieval set. That results in a bigger performance improvement:

<div class="imgcap">
<img src="/assets/retrieval_guided_I2I/I2_quantitative_results_2.png" width="400">
</div>

## Qualitative Results

Here we show qualitative results of the proposed Retrieval Guided Unsupervised Multi-Domain Image to Image Translation method.

<div class="imgcap">
<img src="/assets/retrieval_guided_I2I/I2I_results_1.png" width="600">
</div>

<div class="imgcap">
<img src="/assets/retrieval_guided_I2I/I22_results_2.png" width="600">
</div>

<div class="imgcap">
<img src="/assets/retrieval_guided_I2I/I2I_results_3.png" width="600">
</div>


<span style="color:brown">**This work has been published in ACM MM 2020. Refer to the [paper](https://arxiv.org/abs/2008.04991) to access the full and formal article.**</span>