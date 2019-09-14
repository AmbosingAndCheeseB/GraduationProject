import tensorflow as tf
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

def build_CNN_classifier(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = tf.Variable(tf.truncated_normal(shape= [5, 5, 1, 32], stddev = 5e-2)) # stddev 표준편차
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides = [1, 1, 1, 1], padding= 'SAME')+ b_conv1) #SAME = 제로패딩

    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding= 'SAME')

    W_conv2 = tf.Variable(tf.truncated_normal(shape = [5, 5, 32, 64], stddev = 5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME')

    W_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], stddev = 5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_output = tf.Variable(tf.truncated_normal(shape=[1024,10], stddev=5e-2))
    b_output = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1, W_output) + b_output
    y_pred = tf.nn.softmax(logits)

    return logits, y_pred

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

y_pred, logits = build_CNN_classifier(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
SAVER_DIR = "model"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restored accuracy : %f" % accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))
        sess.close()
        exit()

    for i in range(10000):
        batch = mnist.train.next_batch(50)

        if i % 100 == 0:
            saver.save(sess, checkpoint_path, global_step=i)
            train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y:batch[1]})
            print("Epoch : %d, 정확도 : %f" % (i, train_accuracy))

        sess.run([train_step], feed_dict= {x:batch[0], y:batch[1]})


    print("정확도 : %f" % accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))