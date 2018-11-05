# /usr/bin/env python
# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope("input"):
  x = tf.placeholder(tf.float32, [None, 784], name="x-input")
  x_image = tf.reshape(x, [-1,28,28,1])

with tf.name_scope("conv1"):
  w_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

with tf.name_scope("pool1"):
  h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope("conv2"):
  w_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

with tf.name_scope("pool2"):
  h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc-hidden'):
  w_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

with tf.name_scope('dropoout'):
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc-output'):
  w_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
  y = y_conv

with tf.name_scope('output'):
  y_ = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope("optimizer"):
  loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  optimizer = tf.train.AdamOptimizer(1e-4)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_step = optimizer.minimize(loss, global_step)
  tf.scalar_summary('cross entropy', loss)

with tf.name_scope("accuracy"):
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary('accuracy', accuracy)


def main():
  mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)

  with tf.Session() as session:
    saver = tf.train.Saver()
    writer = tf.train.SummaryWriter('./logs/', graph=session.graph)
    checkpoint = tf.train.get_checkpoint_state('./checkpoints/')

    if checkpoint:
      last_checkpoint = checkpoint.model_checkpoint_path
      saver.restore(session, last_checkpoint)
    else:
      init = tf.initialize_all_variables()
      session.run(init)

    i = global_step.eval(session)
    while i < 20000:
       batch_xs, batch_ys = mnist.train.next_batch(50)
       train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.5})
       i += 1

       if i%100 == 0:
         summary_str, acc = session.run([tf.merge_all_summaries(), accuracy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
         writer.add_summary(summary_str, i)
         saver.save(session, './checkpoints/traindata', global_step = i)
         print("step %d, training accuracy %g" % (i, acc))

    # Test trained model
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    session.close()

if __name__ == "__main__":
    main()
