---
layout: page
title: Hate Speech Labeler
permalink: /hatespic_labeler/
order: 10
---
<!---
<center>
	<object data="http://158.109.9.237:45993" width="600" height="700"> 
	    Your browser doesn’t support the object tag. 
	</object>
</center>
-->

<div class="imgcap">
<a href="http://158.109.9.237:45993"
target="_blank"><img src="/assets/hatespic_labeler/labeler.png" height = "550"></a>
</div>

### What is Hate Speech?
Hate speech is defined as ([Facebook, 2016](https://www.facebook.com/notes/facebook-safety/controversial-harmful-and-hateful-speech-on-facebook/574430655911054/), [Twitter, 2016](https://help.twitter.com/en/rules-and-policies/hateful-conduct-policy)): 

> "Direct and serious attacks on any protected category of people **based on** their race, ethnicity, national origin, religion, sex, gender, sexual orientation, disability or disease."

In the data to be labeled here, which comes from Twitter, it can be summarized as **racism** and **sexism**.

Note that not all critic tweets are Hate Speech:

<span style="color:brown">*"Rachel, you are stupid as all women are."*</span>     
Is Hate Speech, because it attacks a group of people based on gender.


<span style="color:brown">"Rachel, you are stupid."*</span>     
Is not Hate Speech.

### Rules to label Hate Speech
Extracted from Wassem article "[Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter](https://www.aclweb.org/anthology/N/N16/N16-2013.pdf)".


A tweet is offensive if it

1. uses a sexist or racial slur.
2. attacks a minority.
3. seeks to silence a minority.
4. criticizes a minority (without a well founded argument).
5. promotes, but does not directly use, hate speech or violent crime.
6. criticizes a minority and uses a straw man argument.
7. blatantly misrepresents truth or seeks to distort views on a minority with unfounded claims.
8. shows support of problematic hash tags. E.g. “#BanIslam”, “#whoriental”, “#whitegenocide”
9. negatively stereotypes a minority.
10. defends xenophobia or sexism.
11. contains a screen name that is offensive, as per the previous criteria, the tweet is ambiguous (at best), and the tweet is on a topic that satisfies any of the above criteria.

### Examples

<div class="imgcap"><img src="/assets/hatespic_labeler/hate_button.png" height = "30"></div>

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

<div class="imgcap"><img src="/assets/hatespic_labeler/nothate_button.png" height = "30"></div>

<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/hatespic_labeler/nothate_1.png" height = "330">
	</div>
	<div style="display:inline-block; margin-left: 10px;">
		<img src="/assets/hatespic_labeler/nothate_2.png" height = "330">
	</div>
	<div class="thecap">
	These posts are not Hate Speech, because they are not attacks on a protected category of people or based on race, gender, religion, etc..
	</div>
</div>




### The project
There are several works in automatic Hate Speech detection, but all of them use only textual data. A recent survey of them can be found here ([Schmidt, 2017](http://www.aclweb.org/anthology/W17-1101)). With this project we pretend to extend that work to a multi-modal (text and image) analysis. Our goal is to **build a model to automatically detect Hate Speech exploiting both textual and visual data**.
Notice that in some text + image publications nor text or image separately are Hate Speech, but the combination of them is (see the examples above). 

As the Hate Speech detection in multi-modal publications problem has not been adressed yet, there are not available annotated datasets. This Hate Speech Labeler will help us to create a dataset, using the human annotations generated here and exploiting the knwoledge extracted from existing text-only annotated datasets.

### Data selection
The data to be labeled are tweets suspected of containing Hate Speech. In order to make the selection, we have **selected tweets containing hate words** ([see Hatebase.org](https://www.hatebase.org/)) and we have used a **machine learning model trained on existing text-only databases to filter them**. To ensure the dataset diversity, some random tweets will also appear.

Notice that, despite the filtering, the presence of Hate Speech might be small. So it will be normal to label most tweets as Not Hate.
