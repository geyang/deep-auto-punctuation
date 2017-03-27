# Learning Auto-Punctuation by Reading Engadget Articles

## Overview

This project trains a bi-directional GRU to learn how to automatically punctuate a sentence by reading it character by character. The set of operation it learns include:
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

## Setup and Considerations

The initial setup I began with was a single uni-direction GRU, with input domain [A-z0-9] and output domain of the ops listed above. My hope at that time was to simply train the RNN to learn correcponding operations. A few things jumped out of the experiment:

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
    
    example accuracy output:
    ```
    Key: <nop>	Prec:  98.9%	Recall:  97.8%	F-Score:  98.4%
    Key:   ,	Prec:   0.0%	Recall:   0.0%	F-Score:   N/A
    Key: <cap>	Prec: 100.0%	Recall:  60.0%	F-Score:  75.0%
    Key:   .	Prec:   0.0%	Recall:   0.0%	F-Score:   N/A
    Key:   '	Prec:  66.7%	Recall: 100.0%	F-Score:  80.0%
    ```
    
6. **Hidden Layer initialization**: In the past I've found it was easier for the neural network to generate good results when both the training and the generation starts with a zero initial state. In this case because we are computing time limited, I zero the hidden layer at the begining of each file. 

## Data and Cross-Validation

The entire dataset is composed of around 50k blog posts from engadget. I randomly selected 49k of these as my training set, 50 as my validation set, and around 0.5k as my test set. The training is a bit slow on an Intel i7 desktop, averaging 1.5s/file depending on the length of the file. As a result, it takes about a day to go through the entire training set.

## Todo:
- [ ] add validation (after the first epoch)
- [ ] add test to bottom of the notebook

## Done:
- [x] get data
- [x] change to bi-directional GRU
- [x] add accuracy metric, use precision/recall.
- [x] Add temperature to generator
- [x] add self-feeding generator
- [x] get training to work
- [x] use optim and Adam

## References
1: https://www.aclweb.org/anthology/D/D16/D16-1111.pdf  
2: https://phon.ioc.ee/dokuwiki/lib/exe/fetch.php?media=people:tanel:interspeech2015-paper-punct.pdf  
3: https://en.wikipedia.org/wiki/precision_and_recall  
4: https://en.wikipedia.org/wiki/F1_score  