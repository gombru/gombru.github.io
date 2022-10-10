---
layout: post
comments: true
title: "Learning Users’ Preferred Visual Styles in an Image Marketplace"
excerpt: "After my PhD I joined Shutterstock where I've been working on Recommender Systems. The last year I've been developing Visual Styles RecSys, a model that learns users' visual style preferences transversal to the projects they work on, and which aims to personalise the content served at Shutterstock. It was presented as an oral in ACM RecSys '22 industrial track."
date: 2022-10-10 20:00:00
img: "/assets/acm_recsys.png"
mathjax: false
---

<script type="text/javascript" async
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

I stopped publishing in this blog after defending my PhD and joining Shutterstock. But since we have recently published an article and I gave a talk at ACM RecSys 2022, I wanted to share here some updates and material.  

I spent the last year and a half working on RecSys. It has been an amazing experience during which I learnt a lot from a field that shares many tech with computer vision. I could say I've been working on the intersection of those fields since I've been applying Recommender Systems to images. And in RecSys domain knowledge is a must, so my computer vision (and deep image representation learning) experience has been key there. Actually, my last [ECCV paper](https://gombru.github.io/2020/06/03/LocSens/) where I proposed a model for location-sensitive image retrieval was a RecSys, but by then I didn't know!  

Specifically I've been working in a RecSys that learns users' long term visual style preferences, based on image features of diverse natures. Next I link to the different resources that have been published about that.


**Paper**

Providing meaningful recommendations in a content marketplace is challenging due to the fact that users are not the final content consumers. Instead, most users are creatives whose interests, linked to the projects they work on, change rapidly and abruptly. To address the challenging task of recommending images to content creators, we design a RecSys that learns visual styles preferences transversal to the semantics of the projects users work on. We analyze the challenges of the task compared to content-based recommendations driven by semantics, propose an evaluation setup, and explain its applications in a global image marketplace.  

<div class="imgcap">
<img src="/assets/acm_recsys.png" height="180">
</div>

[[PDF](https://dl.acm.org/doi/pdf/10.1145/3523227.3547382)] [[Slides](https://drive.google.com/file/d/1HCXO7KlkESHQhFeHp57tPHco_siLJ3-M/view?usp=sharing)] [[Poster](https://drive.google.com/file/d/1i3fqZxunmF2usg2Ii4DYQtsnsbl1dCUK/view?usp=sharing)]   

An extended [technical report](https://arxiv.org/abs/2208.10902) with more experiments has also been published.


**Blog Post**

I wrote a [blogpost](https://www.shutterstock.com/blog/data-science-building-content-personalization) where I explain in a less formal way how the proposed RecSys works and how we plan to leverage it at Shutterstock, which might be a more pleasant read!  

<div class="imgcap">
<img src="/assets/recsys_results.png" height="180">
</div>


**Presentation**

The recording of the presentation at ACM RecSys 2022 in Seattle is also available:

<p align="center"><iframe align="middle" width="720" height="405" src="https://www.youtube.com/embed/GVLjwuFvhyY" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe></p>


