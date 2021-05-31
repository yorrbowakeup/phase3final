import os
import csv
import glob
import shutil
import pandas as pd


# tag 를 딕셔너리로 바꿔주는 역할
def tag_to_dict(taglocation):
    my = {}
    with open(taglocation, 'rt', encoding='UTF8') as f:
        readCSV = csv.reader(f)
        for row in readCSV:
            tag = [{row[1]: row[2]}]
            if row[0] in my:
                my[row[0]].append({row[1]: row[2]})
            else:
                my[row[0]] = tag
    return my


default_dic = tag_to_dict('Tag.csv')


# 사진파일 jpg 제외하고 나오는 함수
def deleting_jpg(path):
    image_list = [a for a in os.listdir(path) if a.endswith('jpg')]
    image_existing_list = []
    for J in image_list:
        l = os.path.splitext(J)[0]
        image_existing_list.append(l)
    return image_existing_list

# {id: [tag 의 모음]}
def get_image_tag(id):
    c = []
    s = default_dic.get(id)
    if s:
        d = {}
        v = default_dic[id]
        for x in v:
            for j in x.values():
                c.append(j)
        #for x in c:
        d[id] = c
        return d
    else:
        pass
#print(len(default_dic))
#print(get_image_tag('6127854473'))
"""
j = deleting_jpg('photos_clustered_result/CLUSTER_MEANSHIFT_TEST7/0')

print(j)
k = []
for p in j:
    t = get_image_tag(p)
    print(t)
    k.append(t)
print(k)
"""

#각 아이디에 있는 tag를 리스트로 변환시켜줌


def image_tag_corpus(tag_dic):
    result_corpus = []
    if tag_dic is None:
        return None
    else:
        for i, j in tag_dic.items():
            for k in j:
                result_corpus.append(k)
        return result_corpus


def get_information(i_d):   #0 1 2 3 에서 2는 문자열로 주게 됌
    i_d = int(i_d)
    df = pd.read_csv("Photo.csv", names=['photo_id', 'user_id', 'lat','long', 'taken time', 'group'])
    information = df[df['photo_id'] == i_d].to_dict()
    df_list = [information.get('photo_id'), information.get('user_id'), information.get('lat'), information.get('long')]
    j = []
    for i in df_list:
        for keys, values in i.items():
            j.append(values)
    return j

#dp = pd.read_csv("Tag.csv", names=['id', 'rank', 'name'])
#print(dp['name'].value_counts())
#dx =pd.read_csv("Photo.csv", names=['photo_id', 'user_id', 'lat','long', 'taken time', 'group'])
#print(dx['user_id'].describe())