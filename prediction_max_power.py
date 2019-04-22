import tensorflow as tf
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 랜덤으로 정해도 시드를 정하면 어떤 시드의 가중치가 제일 좋은지 알 수 있음
seed = 1
tf.set_random_seed(seed)

x_data = pd.read_csv('./Train_X.csv')
y_data = pd.read_csv('./Train_Y.csv')
x_test = pd.read_csv('./Test_X.csv')
y_test = pd.read_csv('./Test_Y.csv')

feature = x_data.as_matrix().astype('float32')
label = y_data.as_matrix().astype('float32')

feature_test = x_test.as_matrix().astype('float32')
label_test = y_test.as_matrix().astype('float32')


# 초기값들 설정
learning_rate = 0.0005
num_epoch = 200
batch_size = 100
display_step = 10
hidden1_size = 5
hidden2_size = 5
hidden_depth = 2

dataset = tf.data.Dataset.from_tensor_slices((feature, label))
dataset = dataset.shuffle(buffer_size=100000).batch(batch_size)

iterator = dataset.make_initializable_iterator()
next_data = iterator.get_next()

# 신경망 학습 데이터 변수 생성
x = tf.placeholder(tf.float32, shape=[None, feature.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, label.shape[1]])



# ANN 신경망 모델 구현
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


pre_value = model_ANN(x)

loss = tf.reduce_mean(tf.square(y-pre_value))
train = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

saver_dir = "pre_model"
saver = tf.train.Saver()
ckpt_path = os.path.join(saver_dir, "model")
ckpt = tf.train.get_checkpoint_state(saver_dir)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ckpt_question = input("파라미터를 불러오시겠습니까?(Y/N) ")

    if ckpt_question in "y" or ckpt_question in "Y":
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("저장된 가중치 파라미터가 없습니다.")

    for epoch in range(num_epoch):
        average_loss = 0
        sess.run(iterator.initializer, feed_dict={x: feature})

        while True:
            try:
                next_x, next_y = sess.run(next_data)

                _, current_loss = sess.run([train, loss], feed_dict={x: next_x, y: next_y})

            except tf.errors.OutOfRangeError:
                break

        if epoch % display_step == 0:
            print("Epoch : %d, loss : %f" % ((epoch + 1), current_loss))

        if epoch % 1000 == 0:
            saver.save(sess, ckpt_path, global_step=epoch)

        pre_val, y1 = sess.run([pre_value, y], feed_dict={x: feature_test, y: label_test})

    row = np.arange(1, 101)  # 그래프 가로축 범위
    col1 = []
    col2 = []  # 그래프값 저장할 리스트

    f = open("학습결과.txt", 'a')
    f.write("Seed: %d\n학습률: %0.4f\nEpoch: %d\n은닉층 깊이: %d\n은닉층 노드수: %d, %d\n\n"
            % (seed, learning_rate, num_epoch, hidden_depth, hidden1_size, hidden2_size))

    avg = 0
    for i in range(100):
        print("예측값 : %d, 실제 값 : %d" % (pre_val[i], y1[i]))
        col1.extend(pre_val[i])
        col2.extend(y1[i])  # 그래프 리스트에 결과값 추가
        avg = avg + abs(y1[i] - pre_val[i]) / y1[i]  # 평균 오차율

    print("평균 오차율 : %0.2f %%" % avg)
    f.write("평균 오차율 : %0.2f %%" % avg)
    f.write("\n----------------------------\n")

    f.close()

    # 그래프 그리기 #
    plt.plot(row, col1, 'r', label='Predict')
    plt.plot(row, col2, 'b', label='Real')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Value')
    plt.show()
