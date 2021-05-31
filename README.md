from detectingevent import *
from nltk.tokenize.casual import casual_tokenize
import re
from collections import Counter
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MiniBatchKMeans, KMeans
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
