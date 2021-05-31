from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN, KMeans
import sklearn
import shutil
import numpy as np
import os
from detectingevent import *

token_id = deleting_jpg('0')
matrix = np.load('0_.npy')
matrix = np.array(matrix)
rows, cols = len(matrix), len(matrix)
matrix2 = [[0 for _ in range(rows)] for _ in range(cols)]
for x in range(rows):
    for y in range(cols):
        if x != y :
            cosine_similarity = matrix[x][y]
            cosine_distance = 1 - cosine_similarity
            matrix2[x][y] = cosine_distance
            if cosine_distance < 0:
                print("어 이거 아닌데")
                print(cosine_distance)
                print(x, y)
        else:
            matrix2[x][y] = 0
print(np.array(matrix2).shape)
#print(matrix[2][621], matrix[621][2])
#clustering = SpectralClustering(matrix='precomputed').fit(X)
#clustering = AffinityPropagation(affinity='precomputed').fit(X)
#print(clustering.labels_)
#eigen_values, eigen_vectors = np.linalg.eigh(matrix)
model = KMeans(n_clusters=5, init='k-means++')
model2 = DBSCAN(min_samples=4, eps=0.0005, metric='precomputed')
model3 = AgglomerativeClustering(n_clusters=3,linkage="complete", affinity="cosine", compute_full_tree=True)
XX = model2.fit_predict(matrix)
#XX = model2.fit(matrix2)
print(model2.labels_)
print(token_id)


def image_moving(photo_id, label_, path):
    label = np.unique(label_)
    for k in label:
        os.makedirs('visual111\\' + str(k))
    for name in photo_id:
        shutil.copy2(str(path + str(name) + '.jpg'), 'visual111\\' + str(label_[photo_id.index(name)]))
    return print('success_moving_files')


image_moving(token_id,  model2.labels_, '0/')
