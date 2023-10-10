import numpy as np
import math
def l2norm(vector1, vector2):
    vector = vector1 - vector2
    l2norm = np.linalg.norm(vector)
    return l2norm

def l2normSet(vector1, vector2):
    buffer = []
    for i in range(np.shape(vector1)[0]):
        vector = vector1[i] - vector2[i]
        buffer.append(np.linalg.norm(vector))
    l2norm = np.array(buffer).reshape(-1, 1)
    return l2norm

def cosineSimilarity(vector1, vector2):
    innerProduct = np.dot(vector1, vector2)
    distance1 = np.linalg.norm(vector1)
    distance2 = np.linalg.norm(vector2)
    if distance1 == 0 or distance2 == 0:
        print("Warning! vector with 0 distance")
        return 0
    cosineSimilarity = innerProduct / distance1 / distance2
    return cosineSimilarity

def cosineSimilaritySet(vector1, vector2):
    buffer = []
    for i in range(np.shape(vector1)[0]):
        innerProduct = np.dot(vector1[i], vector2[i])
        distance1 = np.linalg.norm(vector1[i])
        distance2 = np.linalg.norm(vector2[i])
        if distance1 == 0 or distance2 == 0:
            print("Warning! vector with 0 distance")
            buffer.append(0)
        else:
            buffer.append(innerProduct / distance1 / distance2)
    cosineSimilarity = np.array(buffer).reshape(-1, 1)
    return cosineSimilarity

def angularDistance(vector1, vector2):
    innerProduct = np.dot(vector1, vector2)
    distance1 = np.linalg.norm(vector1)
    distance2 = np.linalg.norm(vector2)
    if distance1 == 0 or distance2 == 0:
        print("Warning! vector with 0 distance")
        return 0
    cosineDistance = 1 - (math.acos(innerProduct / distance1 / distance2) / math.pi)
    return cosineDistance

def angularDistanceSet(vector1, vector2):
    buffer = []
    for i in range(np.shape(vector1)[0]):
        innerProduct = np.dot(vector1[i], vector2[i])
        distance1 = np.linalg.norm(vector1[i])
        distance2 = np.linalg.norm(vector2[i])
        if distance1 == 0 or distance2 == 0:
            print("Warning! vector with 0 distance")
            buffer.append(0)
        else:
            buffer.append(1 - (math.acos(innerProduct / distance1 / distance2) / math.pi))
    angularDistance = np.array(buffer).reshape(-1, 1)
    return angularDistance



V1 = np.random.randint(5, size = 4)
V1 = V1.reshape(2, 2)
V2 = np.random.randint(5, size = 4)
V2 = V2.reshape(2, 2)
print(V1)
print(V2)
#print(similarity_tool.l2norm(V1, V2), similarity_tool.cosineSimilarity(V1, V2), similarity_tool.angularDistance(V1, V2))
print(cosineSimilaritySet(V1, V2))
print(l2normSet(V1, V2))
print(angularDistanceSet(V1, V2))

