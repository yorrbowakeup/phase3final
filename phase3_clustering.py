import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN, KMeans
import sklearn
from tfidf import *
from imagemoving import *
from sklearn.metrics.pairwise import cosine_distances

token_id = making_cosine_similarity_matrix('photos_clustered_result/CLUSTER_MEANSHIFT_TEST7/2/')
print(sklearn.__version__)
matrix = np.load('w2v_similarity.npy')
#print(matrix.shape)
#print(matrix[1][1])
matrix = np.array(matrix)
#cosine similarity matrix을 cosine distance matrix로 변경,
rows, cols = len(matrix), len(matrix)
matrix2 = [[0 for _ in range(rows)] for _ in range(cols)]
for x in range(rows):
    for y in range(cols):
        cosine_similarity = matrix[x][y]
        cosine_distance = 1 - cosine_similarity
        matrix2[x][y] = cosine_distance
        if cosine_distance < 0:
            print("어 이거 아닌데")
            print(x, y)
print(np.array(matrix2).shape)

#print(matrix[2][621], matrix[621][2])
#clustering = SpectralClustering(matrix='precomputed').fit(X)
#clustering = AffinityPropagation(affinity='precomputed').fit(X)
#print(clustering.labels_)
#eigen_values, eigen_vectors = np.linalg.eigh(matrix)
model = KMeans(n_clusters=7, init='k-means++')
model2 = DBSCAN(min_samples=4, eps=0.005, metric='cosine')
model3 = AgglomerativeClustering(n_clusters=5 ,linkage="complete", affinity="cosine", compute_full_tree=True)
XX = model3.fit_predict(matrix2)
#XX = model2.fit(matrix2)
print(model3.labels_)
print(token_id)
#https://medium.com/web-mining-is688-spring-2021/how-dishes-are-clustered-together-based-on-the-ingredients-3b357ac02b26
image_moving(token_id, model3.labels_, 'photos_clustered_result/CLUSTER_MEANSHIFT_TEST7/2/')

