---
layout: post
comments: true
title:  "What do people think about Barcelona?"
excerpt: "A joint image and text embedding is trained using Instagram data related with Barcelona. It is shown how the embedding can be used to do interesting social or commercial analysis, which can be extrapolated to other topics."
date:   2018-01-12 20:00:00
img: "/assets/insta_barcelona/breakfast.jpg"
mathjax: false
---

In a former post, [Learning to Learn from Web Data -not published yet-](https://gombru.github.io/), we explain how to embed images and text in the same vectorial space with semantic structure. We compare the performance of different text embeddings, and we prove that Social Media data can be used to learn the mapping of both images and text to this common space. 

## Objective

**This post aims to show that learning this common space from Social Media data has very useful applications. To do so, we will learn the embedding with Instagram posts associated to a specific topic: Barcelona. That is, images with captions where the word “Barcelona” appears.**
Once the embeddings are learnt, we will be able to **infer what people talks about when they use the word “Barcelona”, or what images people relate with Barcelona and another topic. That can lead to social or commercial interesting analysis**. For instance:

 - What are the most common words that appear along with Barcelona?
 - What languages do people use most when they speak about Barcelona?
 - What words do people write along with the word "food" and "Barcelona"?
 - What kind of images do people post when they talk about “healthy” and “Barcelona”?
 - What kind of images do people post when they talk about “beer” and “Barcelona”?
 - What kind of images do people post when they talk about “cerveza” and “Barcelona”?
 - What kind of images do people post when they talk about “healthy”, “restaurant” and “Barcelona”?
 - What kind of images do people post when they talk about “gracia” and “Barcelona”?

**Notice that this kind of analysis could be applied to any other concept instead of Barcelona if sufficient data can be collected.**

> The code used is available [here](https://github.com/gombru/insbcn).

> For a more detailed explanation of the embeddings learnt, please refer [here -not published yet-](https://gombru.github.io/) or [here](https://gombru.github.io/2017/06/30/learning_from_instagram/).

## Data adquisition

To download the images from Instagram I used [InstaLooter](https://github.com/althonos/InstaLooter), a simple python script that parses the Instagram web without the need API access (the instagram API is only available for approved apps). You can download images quite fast with that. I searched for the word “barcelona” and downloaded 623K images and captions.

## Dataset filtering

- **Images without a caption or short caption (less than 3 words).**

- **Images with captions in other languages than english, spanish or catalan.**  I used [langdetect](https://pypi.python.org/pypi/langdetect?), a python language detection library ported from Google's language-detection. I discarded posts that had 0 probabilities of belonging to one of those languages.

<div class="imgcap">
<img src="/assets/insta_barcelona/languages.png" height="360">
	<div class="thecap">
	Number of posts collected per language.
	</div>
</div>

- **Images from users contributing with a lot of images.** To avoid spam accounts and single users to influence a lot in the embedding learning, I discarded images from users having more than 20 images.

<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/users1.png" height = "280">
	</div>
	<div style="display:inline-block; margin-left: 10px;">
		<img src="/assets/insta_barcelona/users2.png" height = "280">
	</div>
	<div class="thecap">
	Number of posts of top contributing users. User with most posts: 2374
	</div>
</div>


- **Images containing other cities names in their captions.**  This kind of images tend to be spam.

```
Discards -->
No captions: 2122 
Short caption: 27241 
Language: 37026
User: 161333 
City: 70224 
Number of original vs resulting elements: 325253 vs 623199
```
After the filtering, the dataset was divided in 80% train, 5% validation and 15% test sets.



## Learning the joint embedding

A [Word2Vec](https://code.google.com/archive/p/word2vec/) representation for words is learned using all the text in the training dataset. Notice that a single Word2Vec model is learned for all languages. Then, a regression CNN is trained to map the images to the Word2Vec space.
**For a more detailed explanation of the embeddings learning, please refer [here](https://gombru.github.io/) or [here](https://gombru.github.io/2017/06/30/learning_from_instagram/).**

<div class="imgcap">
<img src="/assets/insta_barcelona/pipeline_training.png" height="200">
	<div class="thecap">
	A regression CNN is trained to map the images to the Word2Vec space. Word2Vec representations of the captions associated to images are used as ground truth.
	</div>
</div>

#### Word2Vec

Word2Vec learns vector representations from non annotated text, where words having similar semantics have similar representations. The learned space has a semantic structure, so we can operate over it (king + woman = queen).
A Word2Vec model is trained from scratch using the [Gensim Word2Vec implementation](https://radimrehurek.com/gensim/models/word2vec.html). A dimensionality of 300 is set for the embedding vectors. We used a window of 8 and do 50 corpus iterations. English, spanish and catalan stop words were removed.
To get the embeddings of the captions, we compute the Word2Vec representation of every word and do the TF-IDF weighted mean over the words in the caption. That's a common practice in order to give more importance in the enconding to more discriminative words. To build our TF-IDF mode, we also use the  [Gensim TF-IDF implementation](https://radimrehurek.com/gensim/models/tfidfmodel.html).

#### Regression CNN

We train a CNN to regress captions Word2Vec embeddings from images. The trained net will let us project any image to the Word2Vec space. The CNN used is a GoogleNet and the framework Caffe.  We train it using Sigmoid Cross Entropy Loss and initializing from the ImageNet trained model.

## Textual analysis using Word2Vec

Word2Vec builds a vectorial space were words having similar semantics are mapped near. So a Word2Vec model trained on Instagram data associated with Barcelona, let’s us do an interesting analysis based solely on textual data. 

### Which words do people associate with “Barcelona” and “ “ :

**Generic:**

 - **food**:  thaifood foodtour eatingout todayfood foodislife smokedsalmon eat degustation foodforthesoul bodegongourmet  
 - **shopping**: shoppingtime shoppingday shopaholic onlineshopping multibrand musthave loveshoes  <span style="color:brown">emporioarmani</span> casualwear fashionday  
 - **beer**: spanishbeer <span style="color:brown">estella</span> <span style="color:brown">desperados</span> beerlover aleandhop beers brewery <span style="color:brown">estrellagalicia</span> <span style="color:brown">mahou</span> goodbee  

**Beer:**

 - **cerveza**: <span style="color:brown">cervezanegra</span> cervezas jarra beertography birra beerlife fresquita birracultura birracooltura lovebeer  
 - **cervesa**: <span style="color:brown">cervesaartesana</span> <span style="color:brown">yobebocraft</span> beernerd <span style="color:brown">idrinkcraft</span> bcnbeer lambicus cerveses instabeer daus <span style="color:brown">cervezaartesana</span>  
 - **estrella**:  <span style="color:brown">spanishbeer</span> cerveza <span style="color:brown">lager</span> damm cnil estrellagalicia estrellabeer cervecera gengibre fritos  
 - **moritz**: <span style="color:brown">moritzbarcelona</span> <span style="color:brown">fabricamoritz</span> beerstagram volldamm craftbeer damm beerxample lovebeer barradebar beerlovers  

**Restaurants:**

 - **sushi + restaurant**: sushibar sushitime japo [gruponomo](https://www.nomomoto.es/) sashimi sushilovers japanesefood bestrestaurant sushiporn comidajaponesa  
 - **healthy + restaurant**: salad eathealthy delicious [flaxkale](http://teresacarles.com/fk/) veggiefood healthyfood [cangambus](http://www.cangambus.cat/la-capella) healthyeating [thegreenspot](http://www.encompaniadelobos.com/the-green-spot/) menjarsaludable  

**Neightbourhoods:**

 - **sants**: barridesants pisapis assajarhostot santsmontjuic <span style="color:brown">inconformistes</span> <span style="color:brown">menueconomico</span> poblesec <span style="color:brown">menubarato</span> santsmontjuc hostafrancs  
 - **gracia**: grcia viladegracia barridegracia barriodegracia farr jardinets grandegracia <span style="color:brown">torrentdelolla</span> <span style="color:brown">hotelcasafuster</span> lanena  
 - **santantoni**: santantoni descobreixbcn <span style="color:brown">vermouthlovers</span> <span style="color:brown">modernism</span> fembarri bcncoffee bcnmoltms <span style="color:brown">vermouthtime</span> mesqhotels larotonda  
 - **badalona**: pontdelpetroli badalonamola lovebadalona santadria badalonacity badalonaturisme <span style="color:brown">escoladevela</span> igbadalona bdn <span style="color:brown">portviu</span>  
 - **sitges**:  igerssitges santperederibes sitgesbeach <span style="color:brown">intadogs</span> garraf <span style="color:brown">gaysitges</span> aiguadol imperfectsalon <span style="color:brown">patinavela</span> visitsitges  

### What atractions do people talk more about? 

We can compare the top visited tourist attractions in Barcelona with its names appearence frequency.

**Most frequent attractions mentioned on Instagram:**  <span style="color:brown"> gaudi, sagradafamilia, barceloneta, parkguell, campnou, tibidabo, sitges, montserrat, gracia, eixample, poblenou, gothic, casabatllo, larambla, raval, lapedrera </span>

**Most visited tourist attractions 2016:**

<div class="imgcap">
<img src="/assets/insta_barcelona/top_attractions.png" height="600">
	<div class="thecap">
	Top visited Barcelona attractions in 2016.
	</div>
</div>

**We can compare the top visited attractions with the most mentioned attractions, which we could see as the most trendy attractions. Because people maybe visits a lot the Museu Picasso but don’t talk about it in Social Media.** A conclusion could be that people talk more about architecture and neighbourhoods than about museums, and that people also post a lot about places near Barcelona (Sitges, Montserrat…).

### Top word in each language

Histograms of the top frequent words in each of the languages.

<div class="imgcap">
<img src="/assets/insta_barcelona/top_words_en.png" height="360">
</div>
<div class="imgcap">
<img src="/assets/insta_barcelona/top_words_es.png" height="360">
</div>
<div class="imgcap">
<img src="/assets/insta_barcelona/top_words_ca.png" height="360">
</div>


## Images associated with text concepts

<div class="imgcap">
<img src="/assets/insta_barcelona/pipeline_retrieval.png" height="160">
	<div class="thecap">
	To use the embedding as an image retrieval by text system, we embed the querying text using the learnt Word2Vec model and we retrieve the nearest images in the joint space.
	</div>
</div>

As the regression CNN has learnt to map images to the Word2Vec space, we can do the same *nearest words* experiment we did with text but with images. That is, **retrieving the images that people associate with the word “Barcelona” and the word “ “:**

#### Generic
**Barcelona**:
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/barcelona_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/barcelona_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/barcelona_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/barcelona_4.jpg" height = "185" width = "185">
	</div>
</div>

**Gaudi**:
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/gaudi_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/gaudi_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/gaudi_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/gaudi_4.jpg" height = "185" width = "185">
	</div>
</div>

#### Food
**Breakfast**: What people have for breakfast in Barcelona? What kind of breakfast people post on Instagram in Barcelona?
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/breakfast_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/breakfast_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/breakfast_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/breakfast_4.jpg" height = "185" width = "185">
	</div>
</div>

**Dinner**: It’s clear that mostly tourist post with this word, and that they always have seafood paella.
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/dinner_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/dinner_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/dinner_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/dinner_4.jpg" height = "185" width = "185">
	</div>
</div>

**Healthy**: What kind of food people think is healthy in Barcelona?
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/healthy_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/healthy_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/healthy_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/healthy_4.jpg" height = "185" width = "185">
	</div>
</div>

**Healthy + Restaurant**: If you have been in Barcelona, you might recognice some places
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/healthy_restaurant_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/healthy_restaurant_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/healthy_restaurant_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/healthy_restaurant_4.jpg" height = "185" width = "185">
	</div>
</div>

**Beer**:
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/beer_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/beer_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/beer_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/beer_4.jpg" height = "185" width = "185">
	</div>
</div>

#### Differences between languages
**Catalonia** (en):
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/catalonia_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/catalonia_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/catalonia_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/catalonia_4.jpg" height = "185" width = "185">
	</div>
</div>

**Cataluña** (es):
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/cataluna_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/cataluna_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/cataluna_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/cataluna_4.jpg" height = "185" width = "185">
	</div>
</div>

**Catalunya** (ca):
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/catalunya_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/catalunya_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/catalunya_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/catalunya_4.jpg" height = "185" width = "185">
	</div>
</div>


#####  Neighbourhoods:
**Poblenou**: Lots of flats being promoted now there
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/poblenou_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/poblenou_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/poblenou_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/poblenou_4.jpg" height = "185" width = "185">
	</div>
</div>

**Poblesec**: A trendy place to have tapas these days
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/poblesec_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/poblesec_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/poblesec_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/poblesec_4.jpg" height = "185" width = "185">
	</div>
</div>

**Rambla**: Touristic Mercat Boqueria
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/rambla_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/rambla_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/rambla_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/rambla_4.jpg" height = "185" width = "185">
	</div>
</div>

**Gracia**: It seems people post a lot of street art photos associated to Gracia
<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/gracia_1.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/gracia_2.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/gracia_3.jpg" height = "185" width = "185">
	</div>
	<div style="display:inline-block; margin-left: 1px;">
		<img src="/assets/insta_barcelona/gracia_4.jpg" height = "185" width = "185">
	</div>
</div>

## TSNE plots

Inspired by Kaparthy who [uses t-SNE to visualize CNN layer features](http://cs.stanford.edu/people/karpathy/cnnembed/), **we use [t-SNE](https://github.com/lvdmaaten/bhtsne/)  to visualize the learnt joint visual and textual embedding**. t-SNE is a non-linear dimensionality reduction method, which we use on our 300 dimensional embeddings to produce 2 dimensional embeddings. 
For each one of the given 400 dimensional visual or textual embeddings, t-SNE computes a 2 dimensional embedding arranging elements that have a similar representation nearby, providing a way to visualize the learnt joint image-text space.

**This representation lets us create a 2-Dimensional image where we can appreciate clusters of the images that have been mapped near in the joint space. In practice, images appearing nearer are images that people post with similar words in Instagram.** We show images of different dimensions that show different semantic granularity. See the full size images to appreciate the results.

<div class="imgcap">
	<div style="display:inline-block">
		<img src="/assets/insta_barcelona/tsne_1k.jpg" height = "290">
	</div>
	<div style="display:inline-block; margin-left: 5px;">
		<img src="/assets/insta_barcelona/tsne_2k.jpg" height = "290">
	</div>
	<div style="display:inline-block; margin-left: 5px;">
		<img src="/assets/insta_barcelona/tsne_4k.jpg" height = "290">
	</div>
</div>

Download [1k](https://github.com/gombru/gombru.github.io/blob/master/assets/insta_barcelona/tsne_1k.jpg), [2k](https://github.com/gombru/gombru.github.io/blob/master/assets/insta_barcelona/tsne_2k.jpg), [4k](https://github.com/gombru/gombru.github.io/blob/master/assets/insta_barcelona/tsne_4k.jpg)

> Off topic: This [Hover Zoom Chrome addon](https://chrome.google.com/webstore/detail/hover-zoom/nonjdcjchghhkdoolnlbekcfllmednbl) shows full size images when hovering on them and it's pretty usefull.

## Conclusion

Social Media data can be used to learn joint image-text embeddings from scratch, and those embeddings can be used to do analysis with high social or commercial value. Notice that this kind of experiments could be applied to any other concept instead of Barcelona, if sufficient data can be collected.

