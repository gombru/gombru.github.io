---
layout: post
comments: true
title:  "Barcelona DeepDream"
excerpt: "Using the Google DeepDream algorithm on models trained with #Barcelona Instagram data to visualize what the users (and the CNN) highlight from the city."
date:   2018-10-10 20:00:00
img: "/assets/BarcelonaDeepDream/intro_wide.jpg"
mathjax: false
---


## Models trained with #Barcelona Instagram data
I research in methods to learn visual features from paired visual and textual data in a self-supervised way. A natural source of this multimodal data is the Web and the social media networks. As an application of my research, I [trained models to learn relations between words and images from Instagram publications related to #Barcelona](https://gombru.github.io/2018/01/12/insta_barcelona/). Using that technology I did an analysis of what relations between words and images do tourists vs locals establish, which was published in the MULA ECCV 2018 workshop, and is available [here](https://gombru.github.io/2018/08/02/InstaBarcelona/).  
In this post I use the Google DeepDream algorithm on those models. **DeepDream magnifies the visual features that a CNN detects on an image**, producing images where the recognized patterns are amplified. Applying it to models trained with #Barcelona Instagram data, we can generate images where the visual features that are more strongly related to most mentioned concepts are amplified, which **shows us in a single canvas the most popular visual features and concepts of #Barcelona**.

## The DeepDream algorithm
The DeepDream algorithm was presented by Google in [this post](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) of its AI Blog in 2015. It magnifies the patterns that a given layer recognizes in an input image, and then generates a new image where those patterns are amplified. The basic idea, as it is explained in their blog, is asking the network: **"Whatever you see there, I want more of it"**. As an example, imagine that we have a CNN trained to recognize only car wheels. When we input an image and use the DeepDream algorithm to amplify what the network sees, it will create an image where any circular shape in the input image has been "converted into" a car wheel (actually, into how the CNN thinks a car wheel looks like).  
We can select any layer and ask the CNN to enhance what it sees. Lower layers will produce lower level patterns, since they are sensible to basic features, but higher layers will produce more sophisticated patterns or whole objects. Applying this algorithm iteratively (to its own output) will result in images where the detected patterns have been more amplified. If we add some zooming and scaling to the iteration process, we can generate images that look pretty good.  
I recommend reading their blog post. They also provide a [IPython notebook](https://github.com/google/deepdream/blob/master/dream.ipynb) based on Caffe with a DeepDream implementation. The code I have used, adapted from that notebook, is available [here](https://github.com/gombru/deepdream).

## Results

To understand what the algorithm is doing, let's inspect what it does in the first iterations.
<div class="imgcap">
	<div style="display:inline-block; margin-left: 2px;">
		<img src="/assets/BarcelonaDeepDream/sky.jpg" height = "200">
	</div>

	<div style="display:inline-block; margin-left: 2px;">
		<img src="/assets/BarcelonaDeepDream/sky-5.jpg" height = "200">
	</div>

	<div style="display:inline-block; margin-left: 2px;">
		<img src="/assets/BarcelonaDeepDream/sky-10.jpg" height = "200">
	</div>

	<div style="display:inline-block; margin-left: 2px;">
		<img src="/assets/BarcelonaDeepDream/sky-15.jpg" height = "200">
	</div>
	<div class="thecap">
	The sky image on the top-left is the input image. In this experiment we are asking the net (which is a GoogleNet) to amplify the features recognized by the inception_5a/3x3 layer. "What do you see in the sky? I want more of it". The top-right image is the result after 10 iterations. It shows that the image sees a dog in the cloud on the left, and some other unrecognizable shapes all over the image. The images on the bot show the result after 15 and 20 iterations, where all the recognized patterns have been amplified. 
	</div>
</div>

### Hallucinating a city

Can you recognize anything in the above results?  

<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/BarcelonaDeepDream/sagrada_familia.jpg" height = "150">
	</div>
	<div style="display:inline-block; margin-left: 10px;">
		<img src="/assets/BarcelonaDeepDream/font_magica.jpg" height = "150">
	</div>

</div>

Let's continue iterating:  

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/city.gif">
	<div class="thecap">
	3000 iterations of the previous experiment with a step of 10.
	</div>
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/city.jpg">
</div>


Interesting right? Looks like a city made of deformed Barcelona top tourist attractions with some random dogs (inherited from the ImageNet pretraining) in it.  

It's even more hallucinatory seeing the iterative process in the opposite direction:  

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/city-reverse.gif">
	<div class="thecap">
	3000 iterations of the previous experiment with a step of 10.
	</div>
</div>


### Using random noise
We don't even need to use an input image. We can use random noise as the starting point, and ask the CNN to amplify what it sees on it.  

<div class="imgcap">
	<div style="display:inline-block; margin-left: 2px;">
		<img src="/assets/BarcelonaDeepDream/noise-5.jpg" height = "200">
	</div>

	<div style="display:inline-block; margin-left: 2px;">
		<img src="/assets/BarcelonaDeepDream/noise-10.jpg" height = "200">
	</div>

	<div style="display:inline-block; margin-left: 2px;">
		<img src="/assets/BarcelonaDeepDream/noise-15.jpg" height = "200">
	</div>

	<div style="display:inline-block; margin-left: 2px;">
		<img src="/assets/BarcelonaDeepDream/noise-35.jpg" height = "200">
	</div>
	<div class="thecap">
	Images at 10, 50, 150 and 350 iterations using random noise as the starting point.
	</div>
</div>

### Guided Hallucinations
Instead of asking the net to amplify what a layer recognizes on an image, we can guide the hallucinations. We do that by using a guide image and asking the net to amplify in the input image what it sees in that guide image.

#### Hamburguer-Guided Hallucination
Using this technique, we can ask the CNN to amplify the *hamburger* recognitions. -Actually, since we have trained the CNN to embed images in a semantic space, we will are asking to amplify *food* recognitions-. 

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/hamburger_initial.jpg"  height = "150">
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/hamburger.gif">
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/hamburger.jpg">
</div>

#### Gaudí-Guided Hallucination
Let's see now what the CNN hallucinates when we ask it to amplify Gaudi related stuff, using as the guide image his dragon statue.

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/gaudi_initial.jpg"  height = "150">
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/gaudi.gif">
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/gaudi.jpg">
</div>

#### Sunset-Guided Hallucination

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/sunset_initial.jpg"  height = "150">
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/sunset.gif">
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/sunset.jpg">
</div>


### Amplifying different layers
Amplifying different layers and using different input images result in really cool images.  

**The Barcelona Islands**   
<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/islands.gif">
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/islands.jpg">
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/islands_2.jpg">
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/islands_3.jpg">
</div>

**The Football Club Barcelona Detector**   
<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/fcb.gif">
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/fcb.jpg">
</div>


**The spacecraft control center**   
<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/spacecraft-reverse.gif">
</div>

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/spacecraft.jpg">
</div>


### Bigger images
The size of the generated images is only limited by your GPU memory. With a Titan X 12GB I've been able to generate 3500x1800 images, as the one shown below. [This DeepDream implementation](https://github.com/crowsonkb/deep_dream) allows to generate bigger images and to use multiple GPU's.

<div class="imgcap">
	<img src="/assets/BarcelonaDeepDream/deepDream_barcelona_islands_big.jpg">
</div>





