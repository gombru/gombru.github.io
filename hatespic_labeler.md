---
layout: page
title: HateSPic Labeler
permalink: /hatespic_labeler/
order: 10
---

<center>
	<object data="https://pytorch.org/tutorials/beginner/data_loading_tutorial.html" width="600" height="700"> 
	    Your browser doesn’t support the object tag. 
	</object>
</center>

### What is Hate Speech?
Hate speech is defined as ([Facebook, 2016](https://www.facebook.com/notes/facebook-safety/controversial-harmful-and-hateful-speech-on-facebook/574430655911054/), [Twitter, 2016](https://help.twitter.com/en/rules-and-policies/hateful-conduct-policy)): 

> "Direct and serious attacks on any protected category of people **based on** their race, ethnicity, national origin, religion, sex, gender, sexual orientation, disability or disease.

In the data to be labeled here, which comes from Twitter, it can be summarized as **racism** and **sexism**.

Note that not all critic tweets are Hate Speech:

<span style="color:brown">*"Rachel, you are stupid as all women are."*</span>     
Is Hate Speech, because it attacks a group of people based on gender.


<span style="color:brown">"Rachel, you are stupid."*</span>     
Is not Hate Speech.

### Examples

<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/hatespic_labeler/hate_1.png" height = "330">
	</div>
	<div style="display:inline-block; margin-left: 10px;">
		<img src="/assets/hatespic_labeler/hate_2.png" height = "330">
	</div>
	<div class="thecap">
	The first example (that will appear always as the first instance to label as a test) is Hate Speech because is racist. The second one is Hate Speech because is sexist.
	</div>
</div>



### The HateSPic project
There are several works in automatic Hate Speech detection, but all of them use only textual data. A recent survey of them can be found here ([Schmidt, 2017](http://www.aclweb.org/anthology/W17-1101)). With this HateSPic project we pretend to extend that work to a multi-modal (text and image) analysis. Our goal is to build a model to automatically detect Hate Speech exploiting both textual and visual data.
Notice that in some text + image publications nor text or image separately are Hate Speech, but the combination of them is (see the examples above). 

As the Hate Speech detection in multi-modal publications problem has not been adressed yet, there are not available annotated datasets. This HateSPic Labeler will help us to create a dataset, using the human annotations generated here and exploiting the knwoledge extracted from existing text-only annotated datasets.

### Data selection
The data to be labeled are tweets suspected of containing Hate Speech. In order to make the selection, we have selected tweets containing hate words ([see Hatebase.org](https://www.hatebase.org/)) and we have used a machine learning model trained on existing text-only databases to filter them. To ensure the dataset diversity, some random tweets will also appear.

Notice that, despite the filtering, the presence of Hate Speech might be small. So it will be normal to label most tweets as Not Hate.
