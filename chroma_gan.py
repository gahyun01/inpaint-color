import os
from keras.models import load_model
from keras.utils import get_custom_objects
import tensorflow as tf 
from tensorflow.python.client import device_lib

# print(tf.sysconfig.get_build_info()["cuda_version"])
# print(tf.sysconfig.get_build_info()["cudnn_version"])
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices() )

# 사용자 정의 손실 함수 등록
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

model = load_model('ChromaGAN/my_model_discriminator.h5', custom_objects={'wasserstein_loss': wasserstein_loss})
print(model)

# # DIRECTORY INFORMATION
# DATASET = "imagenet" # modify
# TEST_NAME ="test1" # modify
# ROOT_DIR = os.path.abspath('../')
# DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/'+DATASET+'/')
# OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/'+DATASET+'/')
# MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/'+DATASET+'/')
# LOG_DIR = os.path.join(ROOT_DIR, 'LOGS/'+DATASET+'/')

# TRAIN_DIR = "train"
# TEST_DIR = "test"

# # DATA INFORMATION
# IMAGE_SIZE = 224
# BATCH_SIZE = 10


# # TRAINING INFORMATION
# PRETRAINED = "my_model_colorization.h5" 
# NUM_EPOCHS = 5