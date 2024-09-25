import matplotlib.pyplot as plt
import numpy as np
import Softmax as SFT

sft = SFT.SoftmaxClassifier(5, 10)

ndata = 1000
x_train = np.random.rand(ndata, 5)
y_train = np.random.randint(0, 2, ndata)
for k in range(5):
    x_train[:, k] = y_train - 0.5 + np.random.randn(ndata)

sft.fit(x_train, y_train, epochs=100)

res = sft.predict(x_train)

ires = np.array([res[k, y_train[k]] for k in range(ndata)])
ores = np.array([res[k, (y_train[k] + 1) % 2] for k in range(ndata)])

hy, hx = np.histogram(ires, bins=10000)
htrue = np.cumsum(hy) / ndata
hy, hx = np.histogram(ores, bins=10000)
hfalse = 1.0 - np.cumsum(hy) / ndata
plt.plot(hfalse, htrue)
plt.xlabel("False Rejection Rate")
plt.ylabel("False Aceptation Rate")
plt.show()
