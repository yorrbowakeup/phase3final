from gensim.models import TfidfModel
import gensim.downloader as api
from helping import *
from gensim import corpora, models, similarities
from collections import defaultdict
import gensim
from gensim.models import Word2Vec
from gensim import models
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
import numpy as np
from gensim import parsing
from gensim.parsing.preprocessing import preprocess_documents

"""
print(torch.cuda.get_device_name(0))
"""


def first_step(path):
    j = deleting_jpg(path)
    return j
"""
j = deleting_jpg('photos_clustered_result/CLUSTER_MEANSHIFT_TEST7/0')
k = []
for p in j:
    t = get_image_tag(p)
    k.append(t)
print(k)
print(len(k))
"""


def making_cosine_similarity_matrix(path):
    f = open('text_tokens.txt', 'w', encoding='UTF-8')
    text_tokens = []
    tokens_id = []
    for p in first_step(path):
        x = get_image_tag(p)
        t = image_tag_corpus(get_image_tag(p))
        if t is None:
            pass
        else:
            text_tokens.append(t)
            tokens_id.append(p)
            ff = ''
            for a in t:
                ff = ff + str(a) + ','
            data = str(p)+ ',' + str(ff) + '\n'
            f.write(data)
    f.close()


    print('id_개수', len(tokens_id))
    print(len(text_tokens))
    dict_Los = corpora.Dictionary(text_tokens)
    tokens_cfs = dict_Los.cfs[10]
    print(tokens_cfs)
    # print(dict_Los)
    # lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    # tokenized 된 dict loss 0 cluster의 모든 단어, tag : number이 tokens임
    tokens = dict_Los.token2id
    print(tokens)

    w2v_model = Word2Vec(sentences=text_tokens,
                         min_count=1,
                         window=1,
                         size=500,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,)
    #w2v_model.save('0_cluster')
    #gates_model = Word2Vec.load('0_cluster')
    #print(w2v_model.wv.similarity('seattle', 'burlesque'))
    #print(len(w2v_model.wv.vocab.keys()))
    #print(w2v_model.wv.n_similarity(text_tokens[0], text_tokens[1]))
    matrix = [[0 for _ in range(len(tokens_id))] for _ in range(len(tokens_id))]
    for i in range(len(tokens_id)):
        for j in range(len(tokens_id)):
            c_s = w2v_model.wv.n_similarity(text_tokens[i], text_tokens[j])
            matrix[i][j] = c_s
            matrix[j][i] = c_s
    #f.close()
    #lf.close()
    holy = np.array(matrix)
    np.save('w2v_similarity', holy)
    return tokens_id
#print(making_cosine_similarity_matrix('D:/bigdata/project/photos'))
#print(making_cosine_similarity_matrix('photos_clustered_result/CLUSTER_MEANSHIFT_TEST7/0'))
#print(get_image_tag('8587568157'))

