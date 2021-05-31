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


def clean_non_alpha(text):
    return re.sub('[^a-zA-Z]', ' ', text)


f = open('text_tokens.txt', 'r', encoding='UTF-8')
text_all = {}
while True:
    line = f.readline()
    if not line: break
    items = line.split(',')
    items.remove('\n')
    #items = (" ").join(items)
    #items = casual_tokenize(items, reduce_len=True, strip_handles=True)
    key, values = items[0], items[1:]
    text_all[key] = values
#print(text_all)

# 각 이미지 폴더의 tag의 bag of words를 만들어보자
my_individual_stopwords =[
    'seattle', 'usa', 'washington',
    'unitedstates', 'unitedstatesofamerica',
    'washingtonstate', 'wa', 'flickr', 'continentnorthamerica',
    'statesuswashington', 'uploaded:by=instagram',
    'seattlewa', 'iphone', 'travel', 'seattlewawashingtonstate',
    'downtown', 'd', 'nikon', 'nikond', 'geotagged', 'geolat', 'geolon',
    'canon', 'tagged', 'instagramapp'
]

def making_bag_of_words(name):
    k = deleting_jpg(name)
    j = []
    bb = []
    #print(k)
    for i in k:
        x = text_all.get(i)
        if x is None:
            pass
        else:
            k = [y for y in text_all[i] if y not in my_individual_stopwords]
            for x in k:
                review_text = re.sub("[^a-zA-Z]", "", x)
                if review_text != '':
                    if review_text not in my_individual_stopwords:
                        bb.append(review_text)
            j = j + k
    return bb
# 가장 빈도수가 높은 10개의 문자를 출력 후 선택적으로 stopwords select
all_tag = []
m_fre_tag = []
for i in range(90):
    most_tag = []
    jjj = making_bag_of_words('after_visual/' + str(i))
    xxx = Counter(jjj)
    yyy = xxx.most_common(10)
    if len(yyy) == 0:
        print('#################################')
        print(i)
        m_fre_tag.append([])
    else:
        for m in yyy:
            q = list(m)
            most_tag.append(q[0])
        m_fre_tag.append(most_tag)
    #print(yyy)
    all_tag.append(jjj)
print(m_fre_tag)
print(m_fre_tag[46])
#print(len(all_tag))
"""
w2v_model = Word2Vec(sentences=all_tag,
                         min_count=1,
                         window=1,
                         size=50,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,)
print(w2v_model.wv.most_similar(all_tag[1]))

matrix = [[0 for _ in range(90)] for _ in range(90)]
#print(matrix)
for i in range(90):
    for j in range(90):
        if all_tag[i]:
            if all_tag[j]:
                c_s = w2v_model.wv.n_similarity(all_tag[i], all_tag[j])
                matrix[i][j] = c_s
                matrix[j][i] = c_s
holy = np.array(matrix)
#np.save('w2v_similarity', holy)
matrix2 = [[0 for _ in range(90)] for _ in range(90)]
for x in range(90):
    for y in range(90):
        cosine_similarity = matrix[x][y]
        cosine_distance = 1 - cosine_similarity
        matrix2[x][y] = cosine_distance
        if cosine_distance < 0:
            print("어 이거 아닌데")
            print(x, y)
print(np.array(matrix2).shape)
model2 = DBSCAN(min_samples=2, eps=0.0001, metric='cosine')
model3 = AgglomerativeClustering(n_clusters=30 ,linkage="complete", affinity="cosine", compute_full_tree=True)
model2.fit(matrix2)
print(model2.labels_)


max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(all_tag)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(all_tag,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")
"""
#print(all_tag)
"""

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)


matrix = [[0 for _ in range(90)] for _ in range(90)]
for x in range(90):
    for y in range(90):
        if x == 25 or x==29 or y==25 or y==29:
            pass
        else:
            pp = jaccard_similarity(m_fre_tag[x], m_fre_tag[y])
            matrix[x][y] = pp
for x in range(90):
    ccc = []
    for y in range(90):
        if matrix[x][y] > 0.1:
            ccc.append(y)
    print(x)
    print("오케이 폴더에 대해서 neighbor는 밑에 꺼야")
    print(ccc)
"""
print(all_tag)
tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 4000,
    stop_words = 'english'
)
data = []
for g in all_tag:
    k = ' '.join(g)
    data.append(k)
print(data)
tfidf.fit(data)
text = tfidf.transform(data)

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1, 2)
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()


find_optimal_clusters(text, 40)
model = MiniBatchKMeans(n_clusters=20, init_size=1024, batch_size=2048, random_state=20, compute_labels=True)
clusters = model.fit_predict(text)
print(model.labels_)
def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=50, replace=False)

    pca = PCA(n_components=2).fit_transform(data[max_items, :].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items, :].todense()))

    idx = np.random.choice(range(pca.shape[0]), size=50, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    plt.show()


plot_tsne_pca(text, clusters)


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))


get_top_keywords(text, clusters, tfidf.get_feature_names(), 10)



