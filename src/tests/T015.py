import Softmax.SoftmaxClassifier as SFT
import numpy as np
import matplotlib.pyplot as plt

sft=SFT(5,2)

xtrain=np.random.randn(100,5)
ytrain=np.zeros([100,2])
ytrain[np.arange(100),(np.random.rand(100)>.5).astype('int')]=1.0

sft.fit(xtrain,ytrain)

res=sft.predict(xtrain)

print(res[0:10])
