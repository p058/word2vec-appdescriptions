import string
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from gensim.models import word2vec  # Force Install numpy(conda install -f numpy) if this scripts hangs when importing this
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors

def tokenize(text):
    tokens = [x.strip() for x in text.split(' ')]
    tokens = [x.lower() for x in tokens if x.isalpha() and len(x) > 2 and x not in StopWords]
    return tokens


def preprocess_text(x):
    x_lower = x.lower()
    x_nopunc = x_lower.translate(None, string.punctuation)
    return x_nopunc


def get_model(app_descriptions, num_features , # Size of vector
    min_word_count = 1,  # minimum frequency to be included for training. (=1 to train on all words)
    context = 10 , # Context window size (number of words to left/right to be used as context)
    downsampling = 1e-2  # proportion to down sample frequently seen words
 ):

    sentences = list(app_descriptions.Words)
    model = word2vec.Word2Vec(sentences, \
                              size=num_features, min_count=min_word_count, \
                              window=context, sample=downsampling, iter=50)
    return model


def get_tfidf(app_descriptions):
    dataWords = app_descriptions.Words.apply(lambda x: list(set(x)))
    idf = Counter([j for sublist in dataWords for j in sublist])
    idfLog = [(j[0], np.log(len(app_descriptions) / float(j[1]))) for j in idf.items()]
    idfLogDict = dict(idfLog)

    app_descriptions['tf'] = app_descriptions.Words.apply(
        lambda x: [(j[0], j[1] / float(len(x))) for j in Counter(x).items()])

    app_descriptions['tf-idf'] = app_descriptions['tf'].apply(
        lambda x: sorted([(j[0], j[1] * idfLogDict[j[0]]) for j in x], key=lambda x: x[1], reverse=True))

    app_descriptions.index = app_descriptions['App Bundle Id']
    apptfidf_dict = app_descriptions['tf-idf'].to_dict()

    return apptfidf_dict

def tf_word(x, vectors, num_features):
    """
    this function takes in the tf-idf scores/word2vec mappings for all words corresponding to an app and returns a weighted average
    of the  app descriptions

    """
    res = np.ones(num_features)

    for word, tf in x:
        res += vectors[word] * tf

    return np.array(res) / float(1 + len(x))


def get_appvec(apptfidf_dict, vectors, num_features):

    app_vector = {}

    for app in apptfidf_dict.keys():
        app_vector[app] = tf_word(apptfidf_dict[app], vectors, num_features)

    return app_vector



if __name__=='__main__':
    app_descriptions = pd.read_csv('app_descriptions_sample.csv')
    StopWords = set(stopwords.words("english"))

    app_descriptions['Description'] = app_descriptions['Description'].apply(lambda x: preprocess_text(x))
    app_descriptions['Words'] = app_descriptions['Description'].apply(lambda x: tokenize(x))


    num_features = 100

    word2vecmodel = get_model(app_descriptions, num_features=num_features)
    ##All words
    words = word2vecmodel.vocab.keys()

    ##word2vec dict
    vectors = {}
    for word in words:
        vectors[word] = word2vecmodel[word]

    tfidf =  get_tfidf(app_descriptions)

    app_vector = get_appvec(tfidf, vectors, num_features)

    datingapp = 'com.myyearbook.m'
    weatherapp1 = 'com.aws.android'
    weatherapp2 = 'com.weather.Weather'

    a = app_vector[datingapp] #dating app
    b = app_vector[weatherapp1] #weather app

    from sklearn.metrics.pairwise import cosine_similarity
    print "similarity between %s, %s is %s" %( datingapp, weatherapp1, cosine_similarity(a.reshape(1,-1),
                                                                                         b.reshape(1,-1))[0,0])

    ## 0.30

    a = app_vector[weatherapp1] #weather app
    b = app_vector[weatherapp2] #weather app

    print "similarity between %s, %s is %s" %( weatherapp1, weatherapp2, cosine_similarity(a.reshape(1,-1), b.reshape(1,
                                                                                                                   -1))[0,0])

    ## 0.98

    uniqueCategories = ['Entertainment', 'Simulation']

    apps_to_plot = list(
        app_descriptions[app_descriptions.Category.isin(uniqueCategories)]['App Bundle Id'])

    X = np.array([app_vector[app] for app in apps_to_plot])

    pca = PCA(n_components=2, whiten=True)

    transformed_vecs = pca.fit_transform(X)

    print pca.explained_variance_ratio_

    app_descriptions.index = app_descriptions['App Bundle Id']
    categories_dict = app_descriptions['Category'].to_dict()
    colorMap = dict(zip(uniqueCategories, mcolors.cnames.keys()[:len(uniqueCategories)]))
    colors = [colorMap[categories_dict[app]] for app in apps_to_plot]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    ax.scatter(transformed_vecs[:, 0], transformed_vecs[:, 1], c=colors, marker='o', s=200, alpha=0.7)
    ax.axhline(y=0.25, color='r')

    ax.set_xlim(-1, 0.6)
    ax.set_ylim(-2, 2)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    fig.savefig("p_components.png")
