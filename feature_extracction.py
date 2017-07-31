#coding=utf-8
import tensorflow as tf
import os
import numpy as np
data=np.genfromtxt("/home/zzq/research/gan_learning/brancer.csv",dtype='unicode',delimiter=',')
pre_data=data[:,2:]
label=data[:,1]
for i in range(len(label)):
    if label[i]=='M':
        label[i]=1
    else:
        label[i]=0
n_input = 30
n_hidden_1 = 30
n_hidden_2 = 30
n_classes = 3
real_input = tf.placeholder(tf.float32, [569, 30], "real_input")
real_labe = tf.placeholder(tf.int32, [569], "real_label")
real_label = tf.one_hot(real_labe, 3)
fake_input = tf.get_variable("fake_input", [569, 30], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.02))
fake_label = tf.constant(2, tf.int32, [569], "fake")
fake_label = tf.one_hot(fake_label, 3)

def multilayer_perceptron(x, name):
    with tf.variable_scope(name):
        weigh1 = tf.get_variable("weight1", [n_input, n_hidden_1], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # Hidden layer with RELU activation
        bias1 = tf.get_variable("bias1", [n_hidden_1], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(0, 0.02))

        layer_1 = tf.add(tf.matmul(x, weigh1), bias1)
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        weigh2 = tf.get_variable("weight2", [n_hidden_1, n_hidden_2], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        bias2 = tf.get_variable("bias2", [n_hidden_2], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(0, 0.02))
        layer_2 = tf.add(tf.matmul(layer_1, weigh2), bias2)
        layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    if name == "generator":
        out_layer = layer_2
    else:
        with tf.variable_scope(name):
            weigh3 = tf.get_variable("weight3", [n_hidden_2, n_classes], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 0.02))
            bias3 = tf.get_variable("bias3", [n_classes], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(0, 0.02))

        out_layer = tf.matmul(layer_2, weigh3) + bias3
    return out_layer


with tf.name_scope("generator"):
    out = multilayer_perceptron(fake_input, "generator")


def discriminator(batch_input, batch_label):
    with tf.name_scope("discriminator"):
        out = multilayer_perceptron(batch_input, "discriminator")
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=batch_label)
        return cross_entropy


with tf.name_scope("real_discriminator"):
    with tf.variable_scope("discriminator"):
        real_predict = discriminator(real_input, real_label)
with tf.name_scope("fake_discriminator"):
    with tf.variable_scope("discriminator", reuse=True):
        fake_predict = discriminator(out, fake_label)
with tf.name_scope("genertor_loss"):
    gen_loss = tf.reduce_sum(tf.log(fake_predict))

with tf.name_scope("discriminator_loss"):
    discrim_loss = tf.reduce_sum(-(tf.log(real_predict) + tf.log(1 - 0.5*fake_predict))) #第二部分爆炸了，因为一开始就被分为fake,然后近似为1
    # discrim_loss = tf.reduce_sum((tf.log(real_predict)))

with tf.name_scope("generator_train"):
    vars_gen = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    gen_optim = tf.train.AdamOptimizer(0.00001, 0.5)
    gradients = gen_optim.compute_gradients(gen_loss, var_list=vars_gen)
    gen_train = gen_optim.apply_gradients(gradients)
with tf.name_scope("discriminator_train"):
    discrim_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
    discrim_optim = tf.train.AdamOptimizer(0.000001, 0.5)
    gradients = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_vars)
    discrim_train = discrim_optim.apply_gradients(gradients)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    fetch={}
    for epoch in range(1000):
        _,_=sess.run([gen_train,discrim_train],feed_dict={
            real_input:pre_data,
            real_labe:label
        })
        d_loss,g_loss=sess.run([discrim_loss,gen_loss],feed_dict={
            real_input:pre_data,
            real_labe:label
        })
        print "d_loss=" +str(d_loss) +'\t'+"g_loss=" +str(g_loss)
