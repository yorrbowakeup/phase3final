import pandas as pd
import os


def deleting_jpg(path):
    image_list = [a for a in os.listdir(path) if a.endswith('jpg')]
    image_existing_list = []
    for J in image_list:
        l = os.path.splitext(J)[0]
        image_existing_list.append(l)
    return image_existing_list

df = pd.read_csv("C:/Users/user/study_coding/project/Photo.csv", names=['photo_id', 'user_id', 'lat','long',
                                                                            'taken_time', 'group'])
# 0 1 2 3 4에서 2(user-id)는 문자열로 주게 됌
def get_information(i_d):
    i_d = int(i_d)
    information = df[df['photo_id'] == i_d].to_dict()
    df_list = [information.get('photo_id'), information.get('user_id'), information.get('lat'), information.get('long'), information.get('taken_time')]
    j = []
    for i in df_list:
        for keys, values in i.items():
            j.append(values)
    return j
# geo-cluster 한 집단 마다- id의 갯수를 비교함. f = |U| / N  (N은 이미지의 개수, U는 unique ID의 개수)
"""
f1 = []
for i in range(74):
    c = deleting_jpg('C:/Users/user/study_coding/project/photos_clustered_result/CLUSTER_MEANSHIFT_TEST7/' + str(i))
    e = []
    print('#########start')
    for d in c:
        x = get_information(d)
        e.append(x[4].split()[0])
    f = set(e)
    print(f)
    print(str(i) + 'f값은 다음과 같음' )
    print(len(f))
    if len(f) < 4:
        f1.append(i)
print(f1)
"""