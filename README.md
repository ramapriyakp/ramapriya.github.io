## Welcome to Ramapriya's Data Scince Project Portfolio
This Repository containing portfolio of data science projects completed by me for academic, self learning purposes to track my progress in AI/Data Science.The projects are from Machine learning,Computer Vision and Natural Language Processing(NLP).

## Computer Vision
[Identify_Blood_Cell_Subtypes_From_Images](https://github.com/ramapriyakp/Portfolio/blob/master/CV/Identify_Blood_Cell_Subtypes_From_Images.ipynb)
<b>Use computer vision techniques to automate methods to detect and classify blood cell subtypes. Trained a convolutional neural network (CNN) for classifying blood cell subtypes. Used Keras Sequential API to define CNN network and classify blood cell subtypes. Achieved Test accuracy of 95% for two class classification by nucleii number and 75% accuracy for four class classification by cell type</b>

[Kangaroo_detect](https://github.com/ramapriyakp/Portfolio/blob/master/CV/Kangaroo_detect.ipynb)
<b>Use Object detection technique to identify presence, location, and type of one or more objects in a given image. Mask R-CNN model pre-fit on MS COCO object detection dataset is used and then trained to the kangaroo dataset. The model is used as-is, wirh  class-specific output layers removed so that new output layers can be defined and trained. Trained Mask R-CNN with resnet101 architecture to achieve Test mAP of 0.967</b>

## Machine Learning
[Credit Fraud Detector](https://github.com/ramapriyakp/Portfolio/blob/master/ML/Credit%20Fraud%20Detector.ipynb)
<b>Credit Card Fraud Detection Problem includes modeling past credit card transactions with the knowledge of the ones that turned out to be fraud.  The main goal is to fit the model either with the dataframes that were undersample and oversample (in order for our models to detect the patterns), and test it on the original testing set.The best classifier is  Logistic Regression with accuracy of 95% for Random UnderSampling and 95% for Oversampling (SMOTE)</b>

[Customers Segmentation (Clustering Model)](https://github.com/ramapriyakp/Portfolio/blob/master/ML/Mall-Customers-Segmentation-Analysis-Clustering-Model.ipynb)
<b>Use Clustering Analysis (KMeans) for very clear insight about the different segments of the customers in the Mall. Then after getting the results we can  accordingly make different marketing strategies and policies to optimize the spending scores of the customer in the Mall.</b>

[Sentiment classification of airlines tweets](https://github.com/ramapriyakp/Portfolio/blob/master/ML/Sentiment%20classification%20of%20airlines%20tweets.ipynb)
<b>This is sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from February of 2015 and analysed to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service"). The accuracy of different classifier models is observed.</b>

## NLP
[Elmo Embeddings](https://github.com/ramapriyakp/Portfolio/blob/master/NLP/Elmo_Embeddings.ipynb)
<b>In this experiment, we build a neural network for sentiment analysis on IMDB movie reviews data. ELMo (Embeddings from Language Models) design uses a deep bidirectional LSTM language model for learning words and their context.The accuracies on unseen texts for ELMo Embeddings is 75%</b>

[Movie_Recommender ](https://github.com/ramapriyakp/Portfolio/blob/master/NLP/Movie_Recommender.ipynb)
<b>We  build a movie recommendation system based on data pertaining to different movies and recommend similar movies of interest! This falls under content-based recommenders. Since focus here is on NLP, we leverage text-based metadata for each movie to try and recommend similar movies based on specific movies of interest. Cosine Similarity is used to calculate a numeric score to denote the similarity between two text documents. Pipeline is Text pre-processing, Feature Engineering, Document Similarity Computation, Find top similar movies, Build movie recommendation function.</b>

[Neural machine Translator (English-German )](https://github.com/ramapriyakp/Portfolio/blob/master/NLP/Neural%20machine%20translator%20English-German%20.ipynb)
<b>We build a seq2seq DL Models for Word Level Language Translation. The objective is to convert a German sentence (sequence of words) to English using a Neural Machine Translation (NMT) system based on word level encoder-decoder models. We use German-English sentence pairs data from http://www.manythings.org/anki/
Sequence-to-Sequence (seq2seq) models are used for a variety of NLP tasks, such as text summarization, speech recognition, language translation, text-to-speech, speech-to-text among others. The model input and output are both sentences. </b>

[Shakespeare_text_generation ](https://github.com/ramapriyakp/Portfolio/blob/master/NLP/Shakespeare_text_generation.ipynb)
<b>We create a language model for generating natural language text by implement and training LSTM.
A trained language model learns the likelihood of occurrence of a word based on the previous sequence of words used in the text. We will first tokenize the seed text, pad the sequences and pass into the trained model to get predicted word.  The multiple predicted words can be appended together to get predicted sequence.</b>

[Spam_filtering_Decision_Tree ](https://github.com/ramapriyakp/Portfolio/blob/master/NLP/Spam_filtering_Decision_Tree.ipynb)
<b>Our goal is to predict whether the new e-mail is spam or not-spam. We use dataset and ML decision tree algorithm and provide the best suited class for the new mail. The algorithm that implements the classification is called a classifier. The accuracy of classifier is 96% and AUC 93%.</b>

[Word_Embeddings ](https://github.com/ramapriyakp/Portfolio/blob/master/NLP/Word_Embeddings.ipynb)
 <b>Word embedding is process of converting each word to a fixed dimensional "word vector". Dimensionality of embedding space (i.e., vector space) is a hyperparameter. The model has and embedding layer used as first layer in network to model text format data.  Test Accuracy of word embedding model is 85%</b>

[Text Classification_Reuter] (https://github.com/ramapriyakp/Portfolio/blob/master/NLP/text_classification_Reuter.ipynb
<b>Reuters-21578 is arguably the most commonly used collection for text classification . The newswire articles is a multi-label problem with highly skewed distribution of documents over categories. The OneVsRestClassifier(LinearSVC) model  metrics is  F1-measure: 0.86 for Micro-average  and F1-measure: 0.44 for Macro-average</b>   
