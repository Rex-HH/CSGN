import tensorflow as tf
from tensorflow.python.client import device_lib

print("TensorFlow version:", tf.__version__)

# 检查可用设备
devices = device_lib.list_local_devices()
for device in devices:
    print(device)

# 检查是否在使用 GPU
print("GPU Available:", tf.config.list_physical_devices('GPU'))