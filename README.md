# Learning Auto-Punctuation by Reading Engadget Articles 

[![DOI](https://zenodo.org/badge/86387963.svg)](https://zenodo.org/badge/latestdoi/86387963) [![](https://img.shields.io/badge/link_on-GitHub-brightgreen.svg?style=flat-square)](https://github.com/episodeyang/deep-auto-punctuation#learning-auto-punctuation-by-reading-engadget-articles)

## Link to Other of my work

[*:star2: Deep Learning Notes*](https://github.com/episodeyang/deep_learning_notes#notes-on-deep-learning-): 
A collection of my notes going from basic multi-layer perceptron to convNet and LSTMs, Tensorflow to pyTorch.

[*:boom: Deep Learning Papers TLDR;*](https://github.com/episodeyang/deep_learning_papers_TLDR#deep-learning-papers-tldr-) 
A growing collection of my notes on deep learning papers! So far covers the top papers from this years ICLR.

## Overview

This project trains a bi-directional GRU to learn how to automatically punctuate a sentence by reading blog posts from Engadget.com character by character. The set of operation it learns include:

```
capitalization: <cap>
         comma:  ,
        period:  .
   dollar sign:  $
     semicolon:  ;
         colon:  :
  single quote:  '
  double quote:  "
  no operation: <nop>
```

### Performance


After 24 epochs of training, the network achieves the following performance on the test-set:

```
    Test P/R After 24 Epochs 
    =================================
    Key: <nop>	Prec:  97.1%	Recall:  97.8%	F-Score:  97.4%
    Key: <cap>	Prec:  68.6%	Recall:  57.8%	F-Score:  62.7%
    Key:   ,	Prec:  30.8%	Recall:  30.9%	F-Score:  30.9%
    Key:   .	Prec:  43.7%	Recall:  38.3%	F-Score:  40.8%
    Key:   '	Prec:  76.9%	Recall:  80.2%	F-Score:  78.5%
    Key:   :	Prec:  10.3%	Recall:   6.1%	F-Score:   7.7%
    Key:   "	Prec:  26.9%	Recall:  45.1%	F-Score:  33.7%
    Key:   $	Prec:  64.3%	Recall:  61.6%	F-Score:  62.9%
    Key:   ;	Prec:   0.0%	Recall:   0.0%	F-Score:   N/A
    Key:   ?	Prec:   0.0%	Recall:   0.0%	F-Score:   N/A
    Key:   !	Prec:   0.0%	Recall:   0.0%	F-Score:   N/A
```
As a frist attempt, the performance is pretty good! Especially since I did not fine tune with a smaller step size afterward, and the Engadget dataset used here is small in size (4MB total).

Double the training gives a small improvement. 

*Table 2. After 48 epochs of training*

```
    Test P/R  Epoch 48 Batch 380
    =================================
    Key: <nop>	Prec:  97.1%	Recall:  98.0%	F-Score:  97.6%
    Key: <cap>	Prec:  73.2%	Recall:  58.9%	F-Score:  65.3%
    Key:   ,	Prec:  35.7%	Recall:  32.2%	F-Score:  33.9%
    Key:   .	Prec:  45.0%	Recall:  39.7%	F-Score:  42.2%
    Key:   '	Prec:  81.7%	Recall:  83.4%	F-Score:  82.5%
    Key:   :	Prec:  12.1%	Recall:  10.8%	F-Score:  11.4%
    Key:   "	Prec:  25.2%	Recall:  44.8%	F-Score:  32.3%
    Key:   $	Prec:  51.4%	Recall:  87.8%	F-Score:  64.9%
    Key:   ;	Prec:   0.0%	Recall:   0.0%	F-Score:   N/A
    Key:   ?	Prec:   5.6%	Recall:   4.8%	F-Score:   5.1%
    Key:   !	Prec:   0.0%	Recall:   0.0%	F-Score:   N/A
```
## Usage

If you feel like using some of the code, you can cite this project via

```bib
@article{deeppunc,
  title={Deep-Auto-Punctuation},
  author={Yang, Ge},
  journal={arxiv},
  year={2017},
  doi={10.5281/zenodo.438358}
  url={https://zenodo.org/record/438358;
       https://github.com/episodeyang/deep-auto-punctuation}
}
```

### To run

First unzip the engagdget data into folder `./engadget_data` by running
```sh
tar -xvzf engadget_data.tar.gz
```

and then open up the notebook [Learning Punctuations by reading Engadget.pynb](Learning%20Punctuations%20by%20reading%20Engadget.ipynb#Learning-Auto-Punctuation-by-Reading-Engadget-Articles), and you can just execute.


To view the reporting, open a `visdom` server by running
```
python visdom.server
```
and then go to http://localhost:8097


### Requirements

```
pytorch numpy matplotlib tqdm bs4
```

## Model Setup and Considerations

The initial setup I began with was a single uni-direction GRU, with input domain [A-z0-9] and output domain of the ops listed above. My hope at that time was to simply train the RNN to learn correcponding operations. A few things jumped out during the experiment:

1. **Use bi-directional GRU.** with the uni-direction GRU, the network quickly learned capitalization of terms, but it had difficulties with single quote. In words like "I'm", "won't", there are simply too much ambiguity from reading only the forward part of the word. The network didn't have enough information to properly infer such punctuations.
    
    So I decided to change the uni-direction GRU to bi-direction GRU. The result is much better prediction for single quotes in concatenations.

    the network is still training, but the precision and recall of single quote is nowt close to 80%.
    
    This use of bi-directional GRU is standard in NLP processes. But it is nice to experience first-hand the difference in performance and training.
    
    A side effect of this switch is that the network now runs almost 2x slower. This leads to the next item in this list:
2. **Use the smallest model possible.** At the very begining, my input embeding was borrowed from the Shakespeare model, so the input space include both capital alphabet as well as lower-case ones. What I didn't realize was that I didn't need the capital cases because all inputs were lower-case. 
    
    So when the training became painfully slow after I switch to bi-directional GRU, I looked for ways to make the training faster. A look at the input embeding made it obvious that half of the embedding space wasn't needed. 

    Removing the lower case bases made the traing around 3x faster. This is a rough estimate since I also decided to redownload the data set at the same time on the same machine.
    
3. **Text formatting**. Proper formating of input text crawed from Engadget.com was crucial, especially because the occurrence of a lot of the puncuation was low and this is a character-level model. You can take a look at the crawed text inside [./engadget_data_tar.gz](./engadget_data_tar.gz). 

4. **Async and Multi-process crawing is much much faster**. I initially wrote the engadget crawer as a single threaded class. Because the python `requests` library is synchronous, the crawler spent virtually all time waiting for the `GET` requests.
    
    This could be made a *lot* faster by parallelizing the crawling, or use proper async pattern. 

    This thought came to me pretty late during the second crawl so I did not implement it. But for future work, parallel and async crawler is going to be on the todo list.

5. **Using Precision/Recall in a multi-class scenario**. The setup makes the reasonable assumption that each operation can only be applied mutually exclusively. The accuracy metric used here are **precision/recall** and the **F-score**, both commonly used in the literature<sup>1,</sup> <sup>2</sup>. The P/R and F-score are implemented according to wikipedia <sup>3,</sup> <sup>4</sup>.
    
    example accuracy report:
    
    ```
    Epoch 0 Batch 400 Test P/R
    =================================
    Key: <nop>	Prec:  99.1%	Recall:  96.6%	F-Score:  97.9%
    Key:   ,	Prec:   0.0%	Recall:   0.0%	F-Score:   N/A
    Key: <cap>	Prec: 100.0%	Recall:  75.0%	F-Score:  85.7%
    Key:   .	Prec:   0.0%	Recall:   0.0%	F-Score:   N/A
    Key:   '	Prec:  66.7%	Recall: 100.0%	F-Score:  80.0%


    true_p:	{'<nop>': 114, '<cap>': 3, "'": 2}
    p:	{'<nop>': 118, '<cap>': 4, "'": 2}
    all_p:	{'<nop>': 115, ',': 2, '<cap>': 3, '.': 1, "'": 3}
    
    400it [06:07,  1.33s/it]
    ```
    
6. **Hidden Layer initialization**: In the past I've found it was easier for the neural network to generate good results when both the training and the generation starts with a zero initial state. In this case because we are computing time limited, I zero the hidden layer at the begining of each file.

7. **Mini-batches and Padding**: During training, I first sort the entire training set by the length of each file (there are 45k of them) and arrange them in batches, so that files inside each batch are roughly similar size, and only minimal padding is needed. Sometimes the file becomes too long. In that case I use `data.fuzzy_chunk_length()` to calculate a good chunk length with heuristics. The result is mostly no padding during most of the trainings.
    
    Going from having no mini-batch to having a minibatch of 128, the time per batch hasn't changed much. The accuracy report above shows the training result after 24 epochs.


## Data and Cross-Validation

The entire dataset is composed of around 50k blog posts from engadget. I randomly selected 49k of these as my training set, 50 as my validation set, and around 0.5k as my test set. The training is a bit slow on an Intel i7 desktop, averaging 1.5s/file depending on the length of the file. As a result, it takes about a day to go through the entire training set.

## Todo:
All done.

## Done:
- [x] execute demo test after training
- [x] add final performance metric
- [x] implement minibatch
- [x] a generative demo
- [x] add validation (once an hour or so)
- [x] add accuracy metric, use precision/recall.
- [x] change to bi-directional GRU
- [x] get data
- [x] Add temperature to generator
- [x] add self-feeding generator
- [x] get training to work
- [x] use optim and Adam

## References
1: https://www.aclweb.org/anthology/D/D16/D16-1111.pdf  
2: https://phon.ioc.ee/dokuwiki/lib/exe/fetch.php?media=people:tanel:interspeech2015-paper-punct.pdf  
3: https://en.wikipedia.org/wiki/precision_and_recall  
4: https://en.wikipedia.org/wiki/F1_score  