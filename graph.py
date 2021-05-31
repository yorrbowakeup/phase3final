import numpy as np
from sklearn.neighbors import NearestNeighbors
from detectingevent import *
from sklearn.cluster import SpectralClustering
import shutil

# targetnumber는 string으로, targetcluster는 int로 받는다
def clustering_visaul(targetnumber, targetcluster):
    model_knn = NearestNeighbors(metric='precomputed', algorithm='brute', n_neighbors=2, n_jobs=-1)
    token_id = deleting_jpg(targetnumber)
    matrix = np.load(targetnumber+'_.npy')
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

    matrix3 = [[0 for _ in range(rows)] for _ in range(cols)]
    for x in range(rows):
        for y in range(cols):
            if x != y:
                if matrix[x][y] < 0.6:
                    matrix3[x][y] = 0
                else:
                    matrix3[x][y] = matrix[x][y]
            else:
                matrix3[x][y] = 1


    nbrs = model_knn.fit(matrix2)
    #distances, indices = nbrs.kneighbors(matrix2)
    #result = nbrs.kneighbors_graph(matrix2).toarray()
    #print(indices)
    #print(result)
    model = SpectralClustering(n_clusters=targetcluster, affinity='precomputed', assign_labels='discretize')
    model.fit(matrix3)
    print(model.labels_)


    def image_moving(photo_id, label_, path):
        label = np.unique(label_)
        for k in label:
            os.makedirs('visual' + targetnumber + '_\\' + str(k))
        for name in photo_id:
            shutil.copy2(str(path + str(name) + '.jpg'), 'visual' + targetnumber + '_\\' + str(label_[photo_id.index(name)]))
        return print('success_moving_files')

    image_moving(token_id, model.labels_, targetnumber + '/')

    return print('이 과정은 성공했어!')


clustering_visaul('7', 10)