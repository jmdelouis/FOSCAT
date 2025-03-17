import time

import tensorflow as tf

# Vérifier si un GPU est disponible
print("GPU détecté :", tf.config.list_physical_devices("GPU"))

device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"


@tf.function
def matmul_benchmark():
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    return tf.reduce_sum(tf.matmul(a, b))


dt = 0
a = 0
for k in range(10):
    # Exécuter le benchmark
    with tf.device(device):
        start = time.time()
        result = matmul_benchmark()
        a += result.numpy()
        end = time.time()
        if k > 2:
            dt += end - start

print(f"Temps de calcul :{dt:.4f} sec", a)
dt = 0
for k in range(10):
    # Exécuter le benchmark
    with tf.device("/CPU:0"):
        start = time.time()
        result = matmul_benchmark()
        a += result.numpy()
        end = time.time()
        if k > 2:
            dt += end - start

print(f"Temps de calcul :{dt:.4f} sec")
