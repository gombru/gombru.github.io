---
layout: post
comments: true
title:  "State of the art of joint image and text embeddings"
excerpt: "A generic state of the art on joint image and text embeddings, gathering the latest publications in the field."
date:   2018-01-22 20:00:00
img: "/assets/soa_joint_embeddings/contrastive_based.png"
mathjax: false
---

**Still working on this...**

Multimodal image and text embeddings are a hot topic. They are used in many areas, such as semantic image retrieval, captioning or classification. Many publications on latest most important computer vision conferences use them. However, states of the art included in those publications often focus on the specific task they are trying to solve (as a general image retrieval state of the art), and lack a general and transversal analysis of latest works on joint image and text embeddings.
I’ve been working with joint image and text embeddings for a year, focusing on semantic image retrieval applications. In this post I review latest works on multimodal embeddings and make a summary of the tasks they are applied to, the different techniques used, the people working on them, and the specific real world problems they have helped to solve.


## Introduction
Deep CNN can take profit of huge annotated datasets to generate powerful image representations. Word embedding methods, such as [Word2Vec](https://code.google.com/archive/p/word2vec/) and [GloVe](https://nlp.stanford.edu/projects/glove/) can take profit of non annotated plain text to generate powerful word embeddings with a semantic structure. LSTM can be trained with non annotated text to produce powerful embeddings of texts of variable lengths.
Those powerful image and text representations have separately brought awesome results on its respective fields. In the computer vision field, image classification, object detection or object segmentation. In the natural language processing field text classification or translation. 
Merging those image and text representations in a joint embedding (or learning both representations at once) allow achieving great results on cross-domain tasks.


## Tasks involving joint image and text embeddings
### Semantic Image Retrieval (or image-sentence matching)
The objective is to retrieve images that are semantically similar to a given query text, or to match image-sentence pairs.
To solve this task, the idea is to learn a joint embedding with semantic structure, where text describing an image is embedded in the same point in the joint space as the image. Once the joint embedding has been learnt, performing multimodal retrieval (image by text, text by image) is straightforward. The query is embedded in the joint space and the closest elements are retrieved.
### Image Captioning
The objective is to describe an image with natural language. It is achieved training an LSTM over image features to produce an image caption. 
### Phrase Localization
The goal of phrase localization is to predict a bounding box in tan image for each entity mention from the caption of the image. A region proposals algorithm is run over the images, and the retrieval is performed over the resulting image patches.
Figure from [Learning Deep Structure-Preserving Image-Text Embeddings](https://arxiv.org/abs/1511.06078), Liwet Wang.
### Image Classification
The semantic structure of the learned joint embedding space has been used to improve image classification results. The idea is to train a CNN to predict the semantic embedding of a label instead a one-hot encoding of the label. This way, the CNN gains robustness against drastical errors.


## Text representations
Before exploring the state of the art on joint image text embedding methods, it’s important to have in mind the characteristics about the text embedding methods they use:
 - **LDA**: Latent Dirichlet Allocation learns latent topics from a collection of text documents and maps words to a vector of probabilities of those topics. It can describe a document by assigning topic distributions to them, which in turn have word distributions assigned.
- **Word2Vec**: Learns relationships between words automatically using a feed-forward neural network. It builds distributed semantic representations of words using the context of them considering both words before and after the target word.
- **GloVe**: It is a count-based model. It learns the word vectors by essentially doing dimensionality reduction on the co-occurrence counts matrix.
-**FastText**: While Word2Vec and GloVe treat each word in a corpus like an atomic entity, FastText treats each word as composed of character ngrams. So the vector of a word is made of the sum of this character ngrams. This is specially useful for morphologically rich languages. This way, it achieves generating better word embedding for rare words and embeddings for out of vocabulary words. 
- **LSTM**: They are given as input CNN image features, and are trained to produce the GT caption word by word. Usually the GT words are one-hot encoded, though other encodings (such as Word2Vec encodings) have been tried in the literature. They can output and encode texts of any length.

> Notice that Word2Vec, GloVe and FastText are word embedding methods, so we’ll need an extra step to produce sentences representations. A simple way to do that is to encode a sentence with the mean of its word vectors, or with the TF-IDF weighted mean. 


## Joint Image and Text Embedding Methods
There is not a task-wise nor a dataset-wise best joint embedding method. Published methods are quite different and most of them claim to be the best in one of the former tasks. However the exposed comparisons are frequently not reliable, because they differ in the datasets used, the splits or the evaluation method. Also, in some works authors claim to be the best when restricted to use certain CNN image features. We could use an standard benchmark and evaluation here! The most used databases are [COCO](http://cocodataset.org/#home), [Fllickr30k](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/) and [Flickr8k](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html). Below we include a image retrieval performance comparison between the listed methods on COCO. Anyway, the objective of this post is not to select the best methods on each task, but to review the different proposed solutions for them.

To summarize the work on joint image and text embeddings, I’ll split the methods in four families. CCA, Contrast based, Regression based and LSTM based.

> The methods described here are mainly methods from the “Computer Vision Community”, which are currently all deep learning based. However, the “Multimedia Community” also works in semantic joint image text embeddings, focusing in the image retrieval task and proposing other kind of approaches. Those works use other datasets, such as [MIRFLICKR](https://dl.acm.org/citation.cfm?id=1743384.1743477) or [NUS-WIDE](https://dl.acm.org/citation.cfm?id=1743384.1743477), and other evaluation metrics.

### CCA
Canonical Correlation Analysis finds linear projections that maximize the correlations between projected vectors from the two views. CCA is hard to scale to large amounts of data.
**[Associating Neural Word Embeddings with Deep Image Representations using Fisher Vectors](https://www.cs.tau.ac.il/~wolf/papers/Klein_Associating_Neural_Word_2015_CVPR_paper.pdf)**. Benjamin Klein et al., Tel Aviv University, CVPR 2015.**
They use Fisher Vectors as sentences representations by pooling the Word2Vec embedding of each word in a sentence, and use CCA to build the joint embedding. They present image retrieval results on Flickr8K, Flickr30K and COCO. 

### Contrast based

<div class="imgcap">
<img src="/assets/soa_joint_embeddings/contrast_based.png">
	<div class="thecap">
	Contrast based methods training pipeline.
	</div>
</div>


In this category I group all methods that use a loss function that, for each image, compares the distances between the inferred image representation with its associated text representation and with a non-matching text representation, maximizing the difference between those distances until a given margin. The objective is to ensure that correct annotations get ranked higher than incorrect ones. This kind of loss is also called **ranking loss** and **[Hingue loss](https://en.wikipedia.org/wiki/Hinge_loss).
Put the loss. Put a figure.
**[DeViSE: A Deep Visual-Semantic Embedding Model] (https://research.google.com/pubs/pub41869.html). Andrea Frome et al., Google, NIPS 2013.**
They train a CNN to predict labels word2vec representations instead of one-hot encoded labels using a ranking loss. This way they achieve a model that generalizes to classes outside of the labeled training set (zero-shot learning), and achieve incorrect predictions to be semantically close to the desired label. They show results on Imagenet classification.
**[Latent Embeddings for Zero-shot Classification](https://arxiv.org/pdf/1603.08895.pdf). Yongqin Xian et al. MPI for Informatics (Germany), CVPR 2016.**
A similar approach to DeVise. They train a CNN to predict labels embeddings using a ranking loss, and report results on zero-shot classification datasets.
**[Learning Deep Structure-Preserving Image-Text Embeddings](https://arxiv.org/abs/1511.06078). Liwei Wang et al., University of Illinois, CVPR 2016.**
They use a bi-directional ranking loss, learning fully connected layers over both fixed image and text embeddings. As the text representation, they user Fisher vectors over Word2Vec.
They present image retrieval results on Flickr30K and COCO, and phrase localization results on Flickr30k Entities.
A journal article of a similar work has also been published: [Learning Two-Branch Neural Networks for Image-Text Matching Tasks](https://arxiv.org/abs/1704.03470). There they also compare to an LSTM encoding in the same pipeline, for which they report slightly worse results.
**[Joint Image-Text Representation by Gaussian Visual-Semantic Embedding](http://www.cs.dartmouth.edu/~chenfang/paper_pdf/GVSE_16.pdf). Zhou Ren et al., University of California, ACM Multimedia 2016.**
They propose a joint embedding where each text concept is mapped to a density distribution in the semantic space, instead to a single point. To do that, they map GloVe text embeddings as Gaussian distributions. They train the model using a ranking loss, and report classification results on MIT Places205 dataset.
You may want to see Zhou Ren’s PhD thesis, [“Joint Image-Text Representation Learning”](https://escholarship.org/uc/item/66f282s6).
**[Beyond instance-level image retrieval: Leveraging captions to learn a global visual representation for semantic retrieval](http://openaccess.thecvf.com/content_cvpr_2017/papers/Gordo_Beyond_Instance-Level_Image_CVPR_2017_paper.pdf). Albert Gordo et al., Xerox Research, CVPR 2017.
They use a bi-directional model trained with a ranking loss. They use tf-idf over Bag of Words to build a text representation. They evaluate using non standard measures on non standard datasets.

### Regression based
In this category I group all methods that train a CNN using a loss to explicitly regress its associated text representation. 

<div class="imgcap">
<img src="/assets/soa_joint_embeddings/regression_based.png">
	<div class="thecap">
	Regression based methods training pipeline.
	</div>
</div>

**[Linking Image and Text with 2-Way Nets](https://www.cs.tau.ac.il/~wolf/papers/capturing-deep-cross.pdf). Avis Eisenschtat et al., Tel Aviv University, CVPR 2017.**
They propose a bi-directional network architecture that employs two tied network channels that project two data sources into a common space using the **Euclidean loss**. As text embeddings, they use Fisher Vectors over Word2Vec. They present image retrieval results on Flickr8k, Flickr30k and COCO.
**[Self-supervised learning of visual features through embedding images into text topic spaces](https://arxiv.org/abs/1705.08631). Lluis Gomez et al., Computer Vision Center, UAB (Spain). CVPR 2017.**
They learn an LDA text representation using wikipedia articles, and then train a CNN to embed the images of those articles to the LDA topic space using a cross entropy loss. They report classification results on PASCAL VOC and retrieval results on the Wikipedia dataset..
**[Cross-Modal Retrieval With CNN Visual Features: A New Baseline](http://ieeexplore.ieee.org/document/7428926/)**. Yunchao Wei et al., IEEE Transactions on Cybernetics, 2017.
They use an LDA as the text encoder and train a CNN using cross entropy loss. They show retrieval results in different multimedia datasets.

### LSTM based

<div class="imgcap">
<img src="/assets/soa_joint_embeddings/lstm_based.png">
	<div class="thecap">
	LSTM based methods pipeline. [Source](http://brain.kaist.ac.kr/research.html).
	</div>
</div>


Here I group methods that are based on inputting CNN image features to an LSTM. They are also “contrast based”, since they use ranking loses.
They are used for image captioning, since the LSTM can produce a caption word by word when image features are inputted, and also for retrieval, using as a common embedding for images and text the latest hidden state of the LSTM.
**[Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models](https://arxiv.org/pdf/1411.2539.pdf)**. Ryan Kiros et al., University of Toronto, TACL 2015.**
They encode sentences using an LSTM over their words Word2Vec representations, and map CNN image features to the same space by a learnable linear projection. They report image retreival results on Flickr8K and Flickr20K.
**[Order-Embeddings of Images and Language](https://arxiv.org/abs/1511.06361). Ivan Vendrov, Ryan Kiros et al., University of Toronto, ICLR 2016.**
They build over above Ruan Kiros work, but taking proffit of Word Net structure and using GRUs to embed sentences. They report image retrieval results on COCO.
**Andrej Karpathy et al. image captioning papers**
[Deep Fragment Embeddings for Bidirectional Image Sentence Mapping](https://cs.stanford.edu/people/karpathy/nips2014.pdf). NIPS 2014.
[Deep Visual-Semantic Alignments for Generating Image Descriptions](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf). CVPR 2015.
DenseCap: Fully Convolutional Localization Networks for Dense Captioning(https://cs.stanford.edu/people/karpathy/densecap.pdf). CVPR 2016.
To describe an image, they use RCNN, which detects objects in the image and produces an embedding for each one of the objects. The representations of a predefined number of objects and of the whole image are concatenated. To describe the sentence, they use an LSTM over its Word2Vec words representations. However, they report to have little change in performance when using random word representations.
They report image retrieval, phrase localization and captioning results on Flickr8K, Flickr30K and COCO.
**[Deep Reinforcement Learning-based Image Captioning with Embedding Reward](https://arxiv.org/abs/1704.03899). Zhou Ren et al. Snap Inc. CVPR 2017.
They address the image captioning task introducing a reinforcement learning reward when generating the captions, that, instead of only taking into account the word to be generated, it takes into account the possible full sentences that could be generated afterwards.
**[Learning Cross-modal Embeddings for Cooking Recipes and Food Images](http://im2recipe.csail.mit.edu/). Amaia Salvador et al., UPC (Spain) and MIT. CVPR 2017.**
They learn a joint embedding of food images, ingredients and text, so they can retrieve a recipe for a given food image. They use an LSTM over word2vec representations to encode text and train the net using a ranking loss. They published the code and the dataset, Recipe1M.


## Results
The table below is a performance comparison of image retrieval, or image-sentence matching in COCO.

<div class="imgcap">
<img src="/assets/soa_joint_embeddings/COCO_table.png">
</div>


