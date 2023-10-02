from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import os

dataNumber = 40   #number should between 0 to 77

os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(temperature = 0)
embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")

#load data
data = pd.read_csv('./20230924溯源_77筆.csv', usecols=["CheckItemName", "StandardValue", "關聯段落(k=1)"], nrows= dataNumber).values
#print(data)
input = data[:, 0] + data[:, 1] + data[:, 2]
#print(input)

vector = np.array(embeddings.embed_documents(input))

#t-SNE
tsne = TSNE(n_components=2, perplexity = 5)
outcome = tsne.fit_transform(vector)
print(outcome)

#plot the graph
type = np.ones((1, dataNumber))


for i in range(dataNumber):
    if data[i, 0] == "混凝土施工作業安全":
        type[0, i] = 0
    elif data[i, 0] == "搖管機具與週邊設施":
        type[0, i] = 1
    elif data[i, 0] == "起重機具作業安全":
        type[0, i] = 2
    elif data[i, 0] == "機具設備":
        type[0, i] = 3
    elif data[i, 0] == "臨水作業":
        type[0, i] = 4
    else:
        type[0, i] = 5
plt.scatter(outcome[:, 0], outcome[:, 1], c = type, cmap = "rainbow")
plt.show()

