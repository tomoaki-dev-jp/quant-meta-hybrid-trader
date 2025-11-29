import tensorflow as tf

devices = tf.config.list_physical_devices("GPU")
device = "/GPU:0" if devices else "/CPU:0"
print("Using:", device)