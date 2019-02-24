import numpy as np
    
datas = np.load("tang.npz")
data = datas['data']
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()

print(type(word2ix))