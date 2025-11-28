import tensorflow as tf

device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
print("Using:", device)

with tf.device(device):
    x = tf.random.normal((1000, 1000))
    y = tf.matmul(x, x)
print("OK")