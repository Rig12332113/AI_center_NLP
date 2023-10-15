import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
dataNumber = 30

V1 = pd.read_csv("vector1_Embedding.csv", usecols=range(1, 1537))
V2 = pd.read_csv("vector2_Embedding.csv", usecols=range(1, 1537))
data = pd.read_csv('./20230924溯源_77筆.csv', usecols=["CheckItemName", "StandardValue", "關聯段落(k=1)", "相關性分數(cosin distance)"], nrows = dataNumber).values
vector1 = np.array(V1)
vector2 = np.array(V2)
vector = np.concatenate((vector1, vector2), axis = 0)

#t-SNE
tsne = manifold.TSNE(n_components=2, perplexity = 10, method = "exact", n_iter = 10000)
outcome = tsne.fit_transform(vector)

#plot the graph
colorType = []
markerType = []
'''
for i in range(dataNumber):
    if data[i, 0] == "混凝土施工作業安全":
        colorType.append('r')
    elif data[i, 0] == "搖管機具與週邊設施":
        colorType.append('g')
    elif data[i, 0] == "起重機具作業安全":
        colorType.append('b')
    elif data[i, 0] == "機具設備":
        colorType.append('orange')
    else:
        colorType.append('grey')
'''
for i in range(dataNumber):
    if data[i, 3] <= 110:
        colorType.append('g')
    elif data[i, 3] > 110 and data[i, 3] <= 150:
        colorType.append('r')
    else:
        colorType.append('grey')

    if data[i, 0] == "混凝土施工作業安全":
        markerType.append('o')
    elif data[i, 0] == "搖管機具與週邊設施":
        markerType.append('x')
    elif data[i, 0] == "起重機具作業安全":
        markerType.append('D')
    elif data[i, 0] == "機具設備":
        markerType.append('*')
    else:
        markerType.append('^')

for i in range(dataNumber):
    #plt.plot(outcome[i, 0], outcome[i, 1], "o", color = colorType[i])
    #plt.plot(outcome[i + dataNumber, 0], outcome[i + dataNumber, 1], "o", color = colorType[i], alpha = 0.3)
    plt.plot(outcome[i, 0], outcome[i, 1], marker = markerType[i], color = colorType[i])
    plt.plot(outcome[i + dataNumber, 0], outcome[i + dataNumber, 1], marker = markerType[i], color = colorType[i], alpha = 0.3)
#plt.savefig("text-embedding-ada-002-sortbydistance-perplexity=5")
plt.show()
