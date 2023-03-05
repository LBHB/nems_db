import numpy as np

import tensorflow as tf # Our main TensorFlow import
import tensorflow_probability as tfp
from nems0.tf.loss_functions import loss_tf_nmse_shrinkage

def eager_function(x):
  result = x ** 2
  print(result)
  return result

def pearson(y_true, y_pred):
    return tfp.stats.correlation(y_true, y_pred, event_axis=None, sample_axis=None)
def pearson2(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=-2, keepdims=True)
    my = tf.reduce_mean(y, axis=-2, keepdims=True)
    xm, ym = x - mx, y - my
    t1_norm = tf.nn.l2_normalize(xm, axis = -2)
    t2_norm = tf.nn.l2_normalize(ym, axis = -2)
    r = tf.reduce_mean(tf.reduce_sum(t1_norm*t2_norm, axis=[-2], keepdims=True))
    #cosine = tf.keras.losses.CosineSimilarity(t1_norm, t2_norm, axis = 0)
    return r
def pearson3(x,y):
    r = [np.corrcoef(x[:,i],y[:,i])[0,1] for i in range(x.shape[1])]
    return np.mean(r)

a = np.zeros((10,20))
a[1,:]=np.arange(20)
x1=a+np.random.randn(10,20)*0.1
y1=a+1+np.random.randn(10,20)*0.1
x = tf.constant(x1)
y = tf.constant(y1)

z=pearson(x,y)
z2=pearson2(x,y)
z3=pearson3(x1,y1)
z,z2,z3

