import glob
import time

# given a string and separator, split the string into token
def tokenize(text, separator):
    return text.split(separator)


# append contents of list2 to list1
def append(list1, list2):
    for item in list2:
        list1.append(item)
    return list1


# load lines of the given text file in an array
def load_text_file(file_path):
    file = open(file_path, 'r')
    return file.readlines()


# load the names of all files matching file_path pattern
# file_path is a unix style string pattern eg - /file/to/directory/*.txt
def load_directory(file_path):
    files = glob.glob(file_path)
    return files


# given file_path, return file name
# example file_path = /file/to/path/filename 1.extension
# return - 'filename 1'
def get_file_name(file_path):
    file_path = file_path.split("/")
    full_name = file_path[len(file_path) - 1]
    name = full_name.split(".")[0]
    return name


def to_float(string_list):
    float_list = []
    for item in string_list:
        float_list.append(float(item))
    return float_list


def to_list(mat):
    ret = []
    for vector in mat:
        temp = []
        for feature in vector:
            temp.append(feature)
        ret.append(temp)
    return ret


def sort(arr, new_object, max_length, sort_key, bit):
    if len(arr) <= 0:
        arr.append(new_object)
        return arr

    i = 0
    while i < max_length and i < len(arr):
        condition = new_object[sort_key] > arr[i][sort_key]
        if bit == 1:
            condition = new_object[sort_key] < arr[i][sort_key]
        if condition:
            temp = arr.copy()
            temp[i] = new_object
            if max_length <= len(arr):
                temp[i + 1: max_length] = arr[i: max_length - 1]
            else:
                temp[i + 1: len(temp)] = arr[i: len(arr)]
            arr = temp
            break
        i = i + 1

    if len(arr) < max_length and i == len(arr):
        arr.append(new_object)

    return arr
