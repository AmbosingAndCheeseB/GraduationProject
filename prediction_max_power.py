import tensorflow as tf
import numpy as np
import pandas as pd


x_data = pd.read_csv('./Train_X.csv')
y_data = pd.read_csv('./Train_Y.csv')
x_test = pd.read_csv('./Test_X.csv')
y_test = pd.read_csv('./Test_Y.csv')

feature = x_data.as_matrix().astype('float32')
label = y_data.as_matrix().astype('float32')

feature_test = x_test.as_matrix().astype('float32')
label_test = y_test.as_matrix().astype('float32')


#초기값들 설정
learning_rate = 0.00005
num_epoch = 10000
batch_size = 100
display_step = 10
hidden1_size = 5
hidden2_size = 5

dataset = tf.data.Dataset.from_tensor_slices((feature, label))
dataset = dataset.batch((batch_size))

iterator = dataset.make_initializable_iterator()
next_data = iterator.get_next()

#신경망 학습 데이터 변수 생성
x = tf.placeholder(tf.float32, shape=[None, feature.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, label.shape[1]])


#ANN 신경망 모델 구현
def model_ANN(x):
    W1 = tf.Variable(tf.random_normal(shape=[feature.shape[1], hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    H1_output = tf.nn.relu(tf.matmul(x, W1) + b1)
    W2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
    H2_output = tf.nn.relu(tf.matmul(H1_output, W2) + b2)
    W_out = tf.Variable(tf.random_normal(shape=[hidden2_size, label.shape[1]]))
    b_out = tf.Variable(tf.random_normal(shape=[label.shape[1]]))
    logit = tf.matmul(H2_output, W_out) + b_out

    return logit

pre_value=model_ANN(x)

loss = tf.reduce_mean(tf.square(y-pre_value))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epoch):
        average_loss = 0
        sess.run(iterator.initializer, feed_dict = {x:feature})

        while True:
            try:
                next_x, next_y = sess.run(next_data)

                _, current_loss = sess.run([train, loss], feed_dict={x:next_x, y:next_y})

            except tf.errors.OutOfRangeError:
                break

        if epoch % display_step == 0:
            print("Epoch : %d, loss : %f" % ((epoch+1), current_loss))

    pre_val, y1 = sess.run([pre_value,y], feed_dict={x:feature_test, y:label_test})
    for i in range(10):

        print("예측값 : %d, 실제 값 : %d" % (pre_val[i], y1[i]))

