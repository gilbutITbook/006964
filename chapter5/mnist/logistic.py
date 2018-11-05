# /usr/bin/env python
# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math

mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)

session = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])

w1 = tf.Variable(tf.random_normal([784, 10]))
b1 = tf.Variable(tf.zeros([10]))
u1 = tf.matmul(x, w1) + b1
y1 = tf.nn.softmax(u1)

y = y1
t = tf.placeholder(tf.float32, [None, 10])

loss = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)

# Training
init = tf.initialize_all_variables()
session.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, t: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, t: mnist.test.labels}) * 100), '%'
