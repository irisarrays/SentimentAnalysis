# Sentiment Analysis for reviews of a restaurant


## Authors
- Xinyu HU(xinyu.hu@student-cs.fr)
- Hugo VANDERPERRE(hugo.vanderperre@student-cs.fr)
- Anurag CHATTERJEE(anurag.chatterjee@student-cs.fr)

## 1. Project Description
This project aims to implement a classifier to predict aspect-based polarities of opinions in reviews of a restaurant.

## 2. Implementation of model

### 2.1 Model selection

The requirement to solve this critical problem is an aspect-based sentiment analysis model.A good amount of research has been done regarding aspect-based models. One can choose either machine learning models  like SVM, logistic regression, random forest, etc.  or deep learning models that are also feasible. We plan to implement a BERT pretrained model because they are predominantly famous in  predicting the opinion on specific aspects relying on the high performance of the bert family for aspect-based sentiment analysis.

### 2.2 Requirement

- transformers==4.17.0
- Pytorch==1.10.0
- spacy==2.2.4

### 2.3 Summary of Final model

All aspects and reviewed objects and completed reviewed sentences are tokenized and fed into the pretrained Bert model. Then, the polarities of the comments are converted to integers, in which way it is easier to compute the loss. The final model consists in a pretrained BERT model (bert base uncased) applied with an Adam optimizer with linear scheduled warm up.

### 2.4 Results

Finally we achieved accuracy **[76.86%, 77.13%, 76.33%,  77.93%,  76.06%]** of deviation set in 5 runs. Mean deviation accuracy is **76.86%** using 332 seconds per run in GPU.

### 2.5 Improvement

Based on the time limit, we could only explore the single Bert pretrained model and Bert pretrained model based on a shallow CNN. Since  CNN performed relatively worse, we removed this part from our final model to avoid loss of accuracy. In future, we can try to implement bert based on LSTM to further customize the bert model and improve the predicted accuracy.

We also tried Bert large uncased pretrained model, expecting to get higher accuracy but the result was the other way round. A suspicious overfitting happened in that case,thus allowing us to ponder that  we can try to combine some based model with bert large to explore more probabilities.
