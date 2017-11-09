---
layout: page
title: Publications
permalink: /publications/
order: 2
---

#### FAST: Facilitated and Accurate Scene Text Proposals through FCN Guided Pruning
Dena Bazazian, **Raul Gomez**, Anguelos Nicolaou, Lluis Gomez, Dimosthenis Karatzas, Andrew D.Bagdanov.   
Pattern Recognition Letters, 2017. [[PDF](http://www.sciencedirect.com/science/article/pii/S0167865517302982)]  

The former DLPR workshop publication lead to this journal publication. We extended our experiments and we improved our algorithm by using the FCN heatmaps to suppress the non-textual regions at the beggining of the text proposals stage, achieving a more efficient pipeline.
<div class="imgcap">
<img src="/assets/publications/fast.jpg" height="300">
</div>

#### Improving Text Proposals for Scene Images with Fully Convolutional Networks
Dena Bazazian, **Raul Gomez**, Anguelos Nicolaou, Lluis Gomez, Dimosthenis Karatzas and Andrew Bagdanov.  
ICPR workshop (DLPR), 2016. [[PDF](https://arxiv.org/abs/1702.05089)]  

This came out from my MS's thesis. It's about how to use a text detection FCN to improve the text proposals algorithm (developed by [Lluis Gomez i Bigorda](http://lluisgomez.github.io/), one of my advisors). The code for the FCN model and training is [here](https://github.com/gombru/TextFCN) and the code for the text proposals pipeline is [here](https://github.com/gombru/TextProposalsInitialSuppression). Watching the FCN detect text in real time is pretty cool.
<div class="imgcap">
<img src="/assets/publications/fcn.gif" height="300">
</div>

