---
layout: post
comments: true
title:  "Selective Text Style Transfer"
excerpt: "A selective style transfer model is trained to learn text styles and transfer them to text instances found in images. Experiments in different text domains (scene text, machine printed text and handwritten text) show the potential of text style transfer in different applications."
date:   2019-05-14 20:00:00
img: "/assets/text_style_transfer/intro.png"
mathjax: false
---

<span style="color:brown">**This is joint work with Ali Furkan Biten and will be published in ICDAR 2019 as "Selective Style Transfer on Text"
<!--[[PDF](#)] (soon available).-->
Refer to the paper to access the full and formal article. Here I explain informally and briefly the experiments conducted and the conclusions obtained.**</span>

## Text Style Transfer
Style Transfer is the task of transferring the *style* of one source image to another keeping its *content*. It's often applied to transfer the style of a painting to a real word image. In that context, the style to be transferred is the line style, the color palette, the brush strokes or the cubist patterns of the painting. In [this blog post](https://gombru.github.io/2019/01/14/miro_styletransfer_deepdream/), where I train a conventional neural style transfer model to transfer Joan Miró painting styles, I explain how style transfer models work. If is the first time you meet neural style transfer, it's a must before you continue this reading.  


In this work we **explore the possibilities of style transfer in the text domain**. We hypothesis that, when working with text images, **style transfer models can learn to transfer a text style (color, text font, etc) to image text instances preserving the original characters**. If that hypothesis is true, text style transfer would have many applications. When working with scene text images, we could generate synthetic images keeping the original text transcription but with a different text style. We could also use it in augmented reality scenarios to stylize any *Arial* text with the text style of the image to augment. Both of these applications would be **very useful as data augmentation techniques**, as shown later in this post. The realistic result we get, prompt that it could also have **artistic applications for graphic design**.
Text style transfer can also be very useful in other text domains: We could train a model to *Arialize* any text found in an image, or to copy the style of a writer in the handwritten text domain.

### Pipeline
As our baseline model we used the neural style transfer model proposed by Google in ["A Learned Representation for Artistic Style"](https://arxiv.org/abs/1610.07629). We initially used their TensorFlow Magenta implementation, which is available [here](https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization). But the code is quite inefficient and outdated, so we reimplemented it in Pytorch.  

Our initial experiments were simple: Train the baseline model with scene text images to see it it could learn to transfer text styles. And yes, it could!

<div class="imgcap">
	<img src="/assets/text_style_transfer/results_all_image.png" width="700">
	<div class="thecap">
	Results of the baseline model transfering scene text styles to scene images. On top, original image is shown. In the middle, 3 of the style images used to train the model are shown. In the bottom, the results of styling the images with those styles.
	</div>
</div>

To make those results useful for any task, we had to be able to **transfer the text style only to textual areas** of the destination image. We called this task **Selective Text Style Transfer**, and came out with two different approaches: A **two-stage** and an **end-to-end** model.

#### Two-Stage model
The proposed two-stage architecture for selective text style transfer stylizes all the image using the baseline model. Then, the probability of each pixel of belonging to a textual area is computed using [TextFCN](https://github.com/gombru/TextFCN). Finally, a **blending weighted by the text probability map of the original and the stylized image** copies style only to textual areas.

<div class="imgcap">
	<img src="/assets/text_style_transfer/2stage.png" width="550">
	<div class="thecap">
	Two-Stage Selective Text Style Transfer pipeline.
	</div>
</div>

#### End-To-End model
The end-to-end model is inspired by the [neural network distillation idea by Hinton](https://arxiv.org/abs/1503.02531), where a single net is trained to perform the tasks of different nets at once. In our case, **we train an image transformation network** (with the same architecture as the original style transfer network) **to perform the tasks of style transfer and text detection at once**, and at the end to transfer the style only to textual areas.  
To train this model we use the COCO-Text dataset (instead of ImageNet), which has annotated text bounding boxes. The training procedure is as follows: We first stylize the whole image with a pre-trained, frozen style transfer model. Then, using the GT annotations, we create an image where only the GT text bounding boxes are stylized. Finally we train the end-to-end model to produce that image given the original image, minimizing a Mean Square Error Loss.  

<div class="imgcap">
	<img src="/assets/text_style_transfer/end2end.png" width="550">
	<div class="thecap">
	End-to-end Selective Text Style Transfer training pipeline.
	</div>
</div>

To achieve a successful training, we had to weight the contributions of textual pixels and non-textual pixels to the loss, empowering the contribution of the first ones. This is because textual areas are very small compared to background in COCO-Text. Check the paper for details!

> The PyTorch code of both the baseline model and the end-to-end model, as well as the trained models, will be available soon.


## Results
### Scene Text Style Transfer
We trained a model using 34 styles from the COCO-Text dataset. Next, some results using both the two-stage and the end-to-end model on COCO-Text images:

<div class="imgcap">
	<img src="/assets/text_style_transfer/sc.png" width="800">
	<div class="thecap">
	Transferring scene text styles (top) to scene text images (left), using both the two-stage and end-to-end models.
	</div>
</div>

The model also allows to interpolate between styles:

<div class="imgcap">
	<img src="/assets/text_style_transfer/sc_weighted.png" width="800">
	<div class="thecap">
	Scene text styles interpolation from "CocaCola" to "Rock".
	</div>
</div>

We also experimented using this model to transfer scene text styles to machine printed and handwritten text images:

<div class="imgcap">
	<img src="/assets/text_style_transfer/mt_ht_sc.png" width="600">
	<div class="thecap">
	Transferring scene text styles to machine text and handwritten text images.
	</div>
</div>

Stylizing machine printed text with scene text styles can have very interesting application in augmented reality scenarios, where we want to generate an image keeping the original scene text style but changing its content. In the following examples, "Icdar" and "Sydney" Arial text has been stylized with scene text stiles and manually inserted in the images:

<div class="imgcap">
	<img src="/assets/text_style_transfer/content_augmentation.png" width="550">
	<div class="thecap">
	Styling machine printed text with a scene text style and replacing the original image text.
	</div>
</div>

Results of the scene text model are appealing. It **successfully learns and transfers the source text style. Not only the color, but also the thickness, the line style, or the background of the characters**. The realistic results suggest that text style transfer can have applications beyond data augmentaion, such as augmented reality or graphic design.

#### Data Augmentation for Scene Text Detection
We used the **two-stage Selective Text Style Transfer pipeline to augment several text detection datasets** (ICDAR 2013, ICDAR 2015 and COCO-Text). Then we trained the [EAST text detector](https://github.com/argman/EAST) on plain and augmented datasets. Training with the augmented datasets, **resulted on a boost in text detection performance** in all the experiments. In this experiment, we used a text style transfer model trained with 96 styles from ICDAR 2015.

<div class="imgcap">
	<img src="/assets/text_style_transfer/table.png" width="450">
	<div class="thecap">
	Results (F-score) of the EAST text detector with different training data, evaluated on ICDAR 2015 Challenge 4 and ICDAR 2013 Challenge 2.
	</div>
</div>

This data augmentation technique has important benefits compared to other methods:
 - The text appears in the same place as in original images, which makes the text position in the image realistic.
 - The text content is keps, which makes the text transcription match the semantics of the scene.
 - It can be used to stylize a large dataset with styles from a smaller dataset you are interested on (as we do with COCO-Text and ICDAR 2015).

 Refer to the paper to get a detailed explanation of these experiments.

### Machine Printed Text Style Transfer
We also explored how text style transfer can learn to transfer machine printed text styles. 
It succesfully copies some style features between machine printed text images:

<div class="imgcap">
	<img src="/assets/text_style_transfer/mt.png" width="600">
	<div class="thecap">
	Transfering styles between machine printed text images. On top, source training styles. On the left, original images.
	</div>
</div>

And also to scene text images where text is not big or difficult:

<div class="imgcap">
	<img src="/assets/text_style_transfer/sc_mt.png" width="600">
	<div class="thecap">
	Transfering machine printed text styles (top) to scene text images (left).
	</div>
</div>

But fails when transferring style to handwritten style images: It transfers correctly some style features, but breaks the content:

<div class="imgcap">
	<img src="/assets/text_style_transfer/hw_mt.png" width="600">
	<div class="thecap">
	Transfering machine printed text styles (top) to handwritten text images (left).
	</div>
</div>

### Handwritten Text Style Transfer
We also trained a model to transfer handwritten styles. It correctly transfers some writer style features between handwritten text images:

<div class="imgcap">
	<img src="/assets/text_style_transfer/hr.png" width="600">
	<div class="thecap">
	Transfering handwritten text styles (top) to handwritten text images (left).
	</div>
</div>

The transference of handwritten style to machine printed text, only works for some machine printed fonts similar to handwritten styles:

<div class="imgcap">
	<img src="/assets/text_style_transfer/mt_hr.png" width="600">
	<div class="thecap">
	Transfering handwritten text styles (top) to machine printed text images (left).
	</div>
</div>

## Conclusions
We have shown that a style transfer model is able to learn text styles as the characters shapes, line style, and colors, and to transfer it to an input text preserving the original characters. We open the field for further research in different directions, such as data augmentation for scene text detection or recognition or handwritten writer identification.
Furthermore, the end-to-end selective style transfer pipeline can be applied in other style transfer tasks besides text. 