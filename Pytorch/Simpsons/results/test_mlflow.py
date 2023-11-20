import torch
import tensorflow as tf

import mlflow
import mlflow.tensorflow
import mlflow.pytorch


conv1_nn_path = 'conv1_nn/try_3/conv1_nn.pt'
conv2_nn_path = 'conv2_nn/try_4/conv2_nn.pt'


conv1_nn = torch.load(conv1_nn_path)
conv2_nn = torch.load(conv2_nn_path)


# conv1_nn registieren
with mlflow.start_run():
    mlflow.log_param("learing_rate", 0.001)
    mlflow.log_param('optimzer','SGD')
    mlflow.log_param('conv_layers', 1)
    mlflow.pytorch.log_model(conv1_nn, artifact_path='models')


# conv2_nn registieren
with mlflow.start_run():
    mlflow.log_param("learing_rate", 0.001)
    mlflow.log_param('optimzer','SGD')
    mlflow.log_param('conv_layers',3)
    mlflow.pytorch.log_model(conv2_nn, artifact_path='models')