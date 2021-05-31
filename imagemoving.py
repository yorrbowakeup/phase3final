import shutil
import numpy as np
import os

#phase3clustering과 imagemoving 2 총 4개의 path를 변경해주어야함
# id는 사진의 id, label는 클러스터링 결과
def image_moving(photo_id, label_, path):
    label = np.unique(label_)
    for k in label:
        os.makedirs('textual_cluster_22\\' + str(k))
    for name in photo_id:
        shutil.copy2(str(path + str(name) + '.jpg'), 'textual_cluster_22\\' + str(label_[photo_id.index(name)]))
    return 'success_moving_files'
