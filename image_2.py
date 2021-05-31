import os
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
from torchvision import models
import pandas as pd
import numpy as np


inputDim = (224,224)
inputDir = "44"
inputDirCNN = "inputImagesCNN"

os.makedirs(inputDirCNN, exist_ok=True)

transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])

for imageName in os.listdir(inputDir):
    I = Image.open(os.path.join(inputDir, imageName))
    newI = transformationForCNNInput(I)
    print(I)
    # copy the rotation information metadata from original image and save, else your transformed images may be rotated
    newI.save(os.path.join(inputDirCNN, imageName))

    newI.close()
    I.close()


class Img2VecResnet18():
    def __init__(self):
        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def getVec(self, img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)

        def copyData(m, i, o): embedding.copy_(o.data)

        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()

        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):
        cnnModel = models.resnet18(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512

        return cnnModel, layer

img2vec = Img2VecResnet18()
allVectors = {}
print("Converting images to feature vectors:")
for image in tqdm(os.listdir("inputImagesCNN")):
    I = Image.open(os.path.join("inputImagesCNN", image))
    vec = img2vec.getVec(I)
    allVectors[image] = vec
    I.close()


def getSimilarityMatrix(vectors):
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / (
                (np.linalg.norm(v, axis=0).reshape(-1, 1)) * ((np.linalg.norm(v, axis=0).reshape(-1, 1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim, columns=keys, index=keys)

    return matrix

def getSimilarityMatrix2(vectors):
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / (
                (np.linalg.norm(v, axis=0).reshape(-1, 1)) * ((np.linalg.norm(v, axis=0).reshape(-1, 1)).T))
    keys = list(vectors.keys())
    return sim

similarityMatrix = getSimilarityMatrix(allVectors)
similarityMatrix2 = getSimilarityMatrix2(allVectors)
print(similarityMatrix.shape)
print(similarityMatrix)
similarityMatrix.to_csv('44_.csv')
np.save('44_', similarityMatrix2)