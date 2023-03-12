# %% [markdown]
# This notebook demonstrates model explainability using [shap](https://shap.readthedocs.io/en/latest/). Firstly we will look at explaining Classical ML models then move on to a RNN model.

# %%
# To install only the requirements of this notebook, uncomment the lines below and run this cell

# ===========================

# !pip install pandas==1.1.5
# !pip install scikit-learn==0.21.3
# !pip install tensorflow==1.14.0
# !pip install numpy==1.19.5
# !pip install matplotlib==3.2.2
# !pip install shap==0.39.0

# ===========================

# %%
# To install the requirements for the entire chapter, uncomment the lines below and run this cell

# ===========================

# try:
#     import google.colab
#     !curl  https://raw.githubusercontent.com/practical-nlp/practical-nlp/master/Ch4/ch4-requirements.txt | xargs -n 1 -L 1 pip install
# except ModuleNotFoundError:
#     !pip install -r "ch4-requirements.txt"

# ===========================

# %% [markdown]
# ## Explaining a logistic regression classifier using shap
# We will be using a similar model to the one on OnePipeline_ManyClassifiers.ipynb notebook.

# %%
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd #to work with csv files

#matplotlib imports are used to plot confusion matrices for the classifiers
import matplotlib as mpl 
import matplotlib.cm as cm 
import matplotlib.pyplot as plt 

#import feature extraction methods from sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words

#pre-processing of text
import string
import re

#import classifiers from sklearn
from sklearn.linear_model import LogisticRegression

#import different metrics to evaluate the classifiers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#import time function from time module to track the training duration
from time import time

# %%
our_data = pd.read_csv("https://query.data.world/s/yd24ckbjzyp7h6zp7bacafpv2lgfkh" , encoding = "ISO-8859-1" )

display(our_data.shape) #Number of rows (instances) and columns in the dataset
display(our_data["relevance"].value_counts()/our_data.shape[0])#Class distribution in the dataset



# convert label to a numerical variable
our_data = our_data[our_data.relevance != "not sure"]
our_data.shape
our_data['relevance'] = our_data.relevance.map({'yes':1, 'no':0}) #relevant is 1, not-relevant is 0. 
our_data = our_data[["text","relevance"]] #Let us take only the two columns we need.
our_data.shape



stopwords = stop_words.ENGLISH_STOP_WORDS
def clean(doc): #doc is a string of text
    doc = doc.replace("</br>", " ") #This text contains a lot of <br/> tags.
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = " ".join([token for token in doc.split() if token not in stopwords])
    #remove punctuation and numbers
    return doc

# %%
import sklearn

#Step 1: train-test split
X = our_data.text #the column text contains textual data to extract features from
y = our_data.relevance #this is the column we are learning to predict. 
print(X.shape, y.shape)
# split X and y into training and testing sets. By default, it splits 75% training and 25% test
#random_state=1 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df=5)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# %%
from sklearn.linear_model import LogisticRegression #import

model = LogisticRegression(class_weight="balanced") #instantiate a logistic regression model
model.fit(X_train_dtm, y_train) #fit the model with training data

#Make predictions on test data
y_pred_class = model.predict(X_test_dtm)

#calculate evaluation measures:
print("Accuracy: ", accuracy_score(y_test, y_pred_class))

# %%
import shap
explainer = shap.LinearExplainer(model, X_train_dtm, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test_dtm)
X_test_array = X_test_dtm.toarray() # we need to pass a dense version for the plotting functions

# %%
from pprint import pprint
pprint(our_data['text'][0])

# %%
shap.initjs()
shap.summary_plot(shap_values, X_test_array, feature_names=vect.get_feature_names())

# %% [markdown]
# The SHAP value plot can further show the positive and negative relationships of the predictors with the target variable.
# 
# This plot is made of all the dots in the train data. It demonstrates the following information:<br>
# 1. Feature importance: Variables are ranked in descending order from the most important decreasing to the least.
# 2. Impact: The horizontal location shows what effect the fature has with respect to prediction (higher or lower prediction).<br>
# 3. Original value: Color shows whether that variable is high (in red) or low (in blue) for that observation.<br>
# 4. Correlation: The presence of the word "dollar" has a high (positive) impact in deciding whether the article is relavent to the USA economy. 

# %%
shap.initjs()

shap.force_plot(
    explainer.expected_value, shap_values[0,:], X_test_array[0,:],
    feature_names=vect.get_feature_names()
)

# %% [markdown]
# Let me describe this elegant plot in great detail:
# The output value is the prediction for that observation (the prediction of the first row in Table B is 6.20).<br>
# 1. The base value: The original [paper](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf) explains that the base value E(y_hat) is “the value that would be predicted if we did not know any features for the current output.” In other words, it is the mean prediction, or mean(yhat).<br>
# 2. Red/blue: Features that push the prediction higher (to the right) are shown in red, and those pushing the prediction lower are in blue.<br>
# 3. Economy: Has a positive impact on deciding if the article is relavent for US Economy. So it pushes the prediction to the right.<br>
# 

# %% [markdown]
# ### Lets move on to the RNN model

# %%
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")
import re
import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import Constant
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()



MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000 
EMBEDDING_DIM = 100 
VALIDATION_SPLIT = 0.2

vocab_size = 20000  # Max number of different word, i.e. model input dimension
maxlen = 1000 # Max number of words kept at the end of each text

# %%
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz", 
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))

    return train_df, test_df

# %%
try:
    from google.colab import files
    
    if not os.path.exists('aclImdb'):
        train,test = download_and_load_datasets()
    else:
        train = load_dataset('aclImdb/train')
        test = load_dataset('aclImdb/test')
    
except ModuleNotFoundError:
    if not os.path.exists('Data/aclImdb'):
        train,test = download_and_load_datasets()
    else:
        train = load_dataset('Data/aclImdb/train')
        test = load_dataset('Data/aclImdb/test')

# %%
train_texts = train['sentence'].values
train_labels = train['polarity'].values
test_texts = test['sentence'].values
test_labels = test['polarity'].values

labels_index = {'pos':1, 'neg':0} 

# %%
#Vectorize these text samples into a 2D integer tensor using Keras Tokenizer
#Tokenizer is fit on training data only, and that is used to tokenize both train and test data.
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts) #Converting text to a vector of word indexes
test_sequences = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#Converting this to sequences to be fed into neural network. Max seq. len is 1000 as set earlier
#initial padding of 0s, until vector is of size MAX_SEQUENCE_LENGTH
trainvalid_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
trainvalid_labels = to_categorical(np.asarray(train_labels))
test_labels = to_categorical(np.asarray(test_labels))

# split the training data into a training set and a validation set
indices = np.arange(trainvalid_data.shape[0])
np.random.shuffle(indices)

trainvalid_data = trainvalid_data[indices]
trainvalid_labels = trainvalid_labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * trainvalid_data.shape[0])

x_train = trainvalid_data[:-num_validation_samples]
y_train = trainvalid_labels[:-num_validation_samples]
x_val = trainvalid_data[-num_validation_samples:]
y_val = trainvalid_labels[-num_validation_samples:]

#This is the data we will use for CNN and RNN training
print('Splitting the train data into train and valid is done')

# %%
batch_size = 64
max_features = vocab_size + 1

#Training an LSTM with embedding on the fly

print("Defining and training an LSTM model, training embedding layer on the fly")
#modified from: 
rnnmodel = Sequential()
rnnmodel.add(Embedding(MAX_NUM_WORDS, 128))
rnnmodel.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
rnnmodel.add(Dense(2, activation='sigmoid'))
rnnmodel.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Training the RNN')
rnnmodel.fit(x_train, y_train,
          batch_size=32,
          epochs=2,
          validation_data=(x_val, y_val))
score, acc = rnnmodel.evaluate(test_data, test_labels,
                            batch_size=32)
print('Test accuracy with RNN:', acc)

# %%
from tensorflow.keras.datasets import imdb
import shap
shap.initjs()

# we use the first 100 training examples as our background dataset to integrate over
explainer = shap.DeepExplainer(rnnmodel, x_train[:20])

#%%
# explain the first 10 predictions
# explaining each prediction requires 2 * background dataset size runs
shap_values = explainer.shap_values(x_val[:2])

import numpy as np
words = imdb.get_word_index()
num2word = {}
for w in words.keys():
    num2word[words[w]] = w
x_val_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), x_val[i]))) for i in range(10)])

# %%
# plot the explanation of the first prediction
# Note the model is "multi-output" because it is rank-2 but only has one column
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_val_words[0])

# %% [markdown]
# Note that each sample is an IMDB review, represented as a sequence of words. This means "feature 0" is the first word in the review, which will be different for difference reviews. <br>
# This means calling summary_plot will combine the importance of all the words by their position in the text. This is likely not what you want for a global measure of feature importance (which is why we have not called summary_plot here). If you do want a global summary of a word's importance you could pull apart the feature attribution values and group them by words.<br><br>
# 
# There are many more amazing plots in SHAP which you can use to explain your model. You can find all of them at [SHAP's](https://github.com/slundberg/shap) github page.


