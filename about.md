---
layout: page
title: About
permalink: /about/
order: 1
---

I'm **Raúl Gómez Bruballa** and I was born in Altafulla and moved to Barcelona to study telecommunications engineering with an specialization in audiovisuals systems at the [Universitat Politècnica de Catalunya](https://telecos.upc.edu/en?set_language=en). After that I did a computer vision master at the [Computer Vision Center](http://www.cvc.uab.es/) (UAB), where I also did my PhD together with [Eurecat](https://eurecat.org/), the Catalonia technological center, under an industrial PhD program.
During my PhD I researched on diverse topics involving visual and textual data, focusing on learning from Web and Social Media data and on applied research. Those topics include multi-modal image retrieval, image tagging, multi-modal hate speech detection or scene text detection.
During my PhD I did research stay at the [Multimedia and Human Understanding Group](http://mhug.disi.unitn.it/), in the University of Trento, and an internship in at [Huawei Ireland Research Center](https://www.linkedin.com/company/huawei-ireland-research-center/mycompany/), in the behavior analysis group.

I'm interested in computer vision, deep learning and image processing, and I like to work on research and development projects that have a direct impact in our society. 
In this personal website I write about my scientific work, either publications, toy experiments or coding stuff.


<div class="imgcap">
<img src="/assets/IMG_20190927_075523.jpg" height="300">
</div>

<div style="display:inline-block; margin-left: 25px;">
[**Curriculum**](https://drive.google.com/file/d/1lVkR3tW6dt93ExcdVR7Jy6NJCQxKR81s/view?usp=sharing)  |  [**Publications**](https://gombru.github.io/publications/)
</div>

### Places you can find me besides in the mountains:

**E-mail**   <a href="mailto:{{ site.email }}">{{ site.email }}</a>  
**Github**   [gombru](https://github.com/gombru)  
**Linkedin**   [Raul Gomez Bruballa](https://www.linkedin.com/in/raulgomezbruballa)  
**Instagram**   [raulgombru](https://www.instagram.com/raulgombru/)  
**Twitter**   [gombru](https://twitter.com/gombru)  
**YouTube**   [gombru](https://www.youtube.com/channel/UC3vvewvchL5Si3bix1Kis6A?view_as=subscriber)  


<div class="imgcap">
	<div style="display:inline-block">
		<script src="https://apis.google.com/js/platform.js"></script>
		<div class="g-ytsubscribe" data-channelid="UC3vvewvchL5Si3bix1Kis6A" data-layout="full" data-count="default"></div>
	</div>
	<div style="display:inline-block; margin-left: 25px;">
<a href="https://twitter.com/gombru?ref_src=twsrc%5Etfw" class="twitter-follow-button" data-show-count="true">Follow @gombru</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
	</div>
</div>



## Portfolio
	

### Education

**2016–2020 PhD in Computer Vision**  
Eurecat and Computer Vision Center, Universitat Autónoma de Barcelona, Exellent Cum Laude. Supervised by Dr. Dimosthenis Karatzas, Dr. Lluis Gómez and Dr. Jaume Gibert.
 - **Research stay in University of Trento**. Working with the Multimedia and Human Understanding Group, lead by Nicu Sebe.
 - **Internship in Huawei Ireland Research Center**. Research intern in Huawei, working in the human behaviour analysis group.  


**2015–2016 Master in Computer Vision**  
Universitat Autónoma de Barcelona. GPA – 8.45.


**2010–2014 Bachelor’s Degree in Telecomunications Engineering**  
Universistat Politécnica de Catalunya, GPA – 6.8. Specialized in Audiovisual Systems

### Experience

**2020-Present Research Intern (PhD).**  
Huawei Ireland Research Center, Dublin. Working on video action recognition for human behaviour analysis.


**2016-2020 Computer Vision Researcher**  
EURECAT Technology Centre, Barcelona. Worked on R&D computer vision consultancy projects and on the PhD research.


**2016-2017 Research assistant**  
Computer Vision Centre, Barcelona. Working with Convolutional Neural Networks in image text understanding.


**2015–2016 Internship, EURECAT, Barcelona.**  
Internship in the context of the master thesis. Worked on text detection.

### Computer Vision R&D

**Image Retrieval**  
I've an extense experience training deep models for image retrieval, using triplet nets architectures and [ranking losses](https://gombru.github.io/2019/04/03/ranking_loss/).
 - Image by text retrieval learning from Social Media data. [[Project](https://gombru.github.io/2018/08/01/learning_from_web_data/)]
 - Face images by attributes retrieval. [[Project](https://gombru.github.io/2020/01/23/face_attributes_retrieval/)]
 - Image retrieval by text and location. [[Project](https://gombru.github.io/2020/06/03/LocSens/)]


<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/about/retrieval_2.png" height = "180">
	</div>
	<div style="display:inline-block; margin-left: 10px;">
		<img src="/assets/about/retrieval_1.png" height = "180">
	</div>
</div>


**Image Tagging and Classification**  
I've also worked in the image tagging tasks (or [Multi Label Classification](https://gombru.github.io/2018/05/23/cross_entropy_loss/)) applied to different scenarios.
 - Tagging of geolocated images. [[Project](https://gombru.github.io/2020/06/03/LocSens/)].
 - Finding the ingredients of food images. [[Project](https://gombru.github.io/2017/07/19/inferring_ingredients/)]

<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/about/tagging_1.png" height = "180">
	</div>
	<div style="display:inline-block; margin-left: 10px;">
		<img src="/assets/about/tagging_2.jpg" height = "180">
	</div>
</div>

**Object Detection**  
I've experience training object detection models (YOLO, Faster-RCNN, Mask-RCNN, etc) in different datasets.
 - I designed a small CNN (based on MobileNet) to embed it in a parking camera for car detection.
 - I trained Mask-RCNN (Detectron 2) for person detection ussing annotated video frames.

 <div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/about/detection_1.jpg" height = "180">
	</div>
	<div style="display:inline-block; margin-left: 10px;">
		<img src="/assets/about/detection_2.png" height = "180">
	</div>
</div>

**Object Segmentation**  
I've trained object segmentation models (FCN, Mask-RCNN) for different tasks.
 - Face and hair segmentation model to embed in a Nvidia Jetson. [[Project](https://gombru.github.io/2018/01/08/face_hair_segmentation/)]
 - Scene Text segmentation to detect text at a pixel level. [[Project](https://gombru.github.io/2018/01/08/face_hair_segmentation/)]
 - Defects in sewers segmentation model for an automatic sewer inspection robot. [[Project](https://eurecat.org/portfolio-items/aerial-robot-for-sewer-inspection/)]

 <div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/facefcn/FaceFCN.gif" height = "180">
	</div>
	<div style="display:inline-block; margin-left: 10px;">
		<img src="/assets/publications/fcn.gif" height = "180">
	</div>
</div>

**Text Detection**  
I've experience in scene text detection and have worked in several projects in the field.
 - I organized the COCO-Text Detection Competition. [[Project](https://rrc.cvc.uab.es/?ch=5&com=evaluation&task=1)]
 - I published a method to improve the former scene text detection pipeline. [[Project](https://github.com/gombru/TextFCN)]
 - Selective Text Style transfer, a model which detects text in an image and then stylizes it. [[Project](https://gombru.github.io/2019/05/14/text_style_transfer/)]

 <div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/publications/coco-text.png" height = "180">
	</div>
	<div style="display:inline-block; margin-left: 10px;">
		<img src="/assets/about/text.png" height = "180">
	</div>
</div>

**Video Understanding**  
I've worked on video action recognition for human behaviour analysis, improving state of the art SlowFast models and training then on several large scale datasets.

 <div class="imgcap">
	<div style="display:inline-block; margin-left: 10px;">
		<img src="/assets/about/ava_demo.gif" height = "140">
	</div>
</div>

**NLP**  
I've also experience with NLP and have trained word representation models (Word2Vec, GLoVe, BERT) and LSTM networks for text understanding, most of the times working in multimodal (images and text) tasks.
 - Training word representation models with Social Media data for an image by text retrieval task. [[Project](ttps://gombru.github.io/2018/08/01/learning_from_web_data/)]
 - Training an LSTM with Twitter data for multimodal hate speech classification. [[Project](https://gombru.github.io/2019/10/09/MMHS/)]

 <div class="imgcap">
	<div style="display:inline-block; margin-left: 10px;">
		<img src="/assets/about/nlp.png" height = "140">
	</div>
</div>

### Other Projects

**Technical Courses Development**  
I've worked developing technical courses to teach machine learning to computer scientist by doing applied projects.
 -  Manning: [Image Retrieval usingTextual Inquiry with Deep Learning](https://www.manning.com/?a_aid=VPNIoT)

**External Writer**  
I've worked as a freelance technical writer and have recognized written communications skills, especially to explain technical concepts in an intuitive way.
 - Neptune Blog Post: [Implementing Content-Based Image Retrieval with Siamese Networks in PyTorch](https://neptune.ai/blog/content-based-image-retrieval-with-siamese-networks)

**SetaMind: Image Classification App**  
I developed this [Android App](play.google.com/store/apps/details?id=gombru.setamind) that, given a photo of a mushroom, recognizes its species. It uses a CNN that runs locally in the phone.

**Social Media Analysis**  
I developed tools that learn from images and associated text, and applied that to Instagram data analysis.
 - I presented an application of this work to #Barcelona Instagram images and tourism in the [TurisTICForum of Barcelona](https://gombru.github.io/2018/01/12/insta_barcelona).

**Blog**  
I have a blog where I explain my PhD work, toy experiments and general machine learning concepts. One of its articles explaining Cross-Entropy Loss is featured in the Deep Learning [fast.ai](https://www.fast.ai/) course and in the [deeplearning.ai](https://www.deeplearning.ai/programs/) Introduction to TensorFlow course. It’s visited by 15k people per month. https://gombru.github.io/. I constantly receive good feedback aplauding my intuitive explanations.  


### Recommendation Letters

Ask for them.