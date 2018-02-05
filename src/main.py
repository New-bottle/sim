import tensorflow as tf
import numpy as np
import pickle

print '------------------LOAD START---------------------'
train_file = '../data/training.pk1'
test_file = '../data/test.pk1'

with open(train_file, 'rb') as f:
	X_tmp = np.array(pickle.load(f))
	Y_data = np.array(pickle.load(f))

with open(test_file, 'rb') as f:
	X_test = np.array(pickle.load(f))

print '-------------------LOAD END----------------------'

input_num = 2+5+4+32+32
learning_rate = 0.99
iterations = 1000
batch_size = 1000
drop_keep_prob = 0.99
hid1 = 100
output = 1

X_data = []
for each in X_tmp:
	tmp5 = np.zeros(5)
	tmp4 = np.zeros(4)
	tmp321 = np.zeros(32)
	tmp322 = np.zeros(32)
	tmp5[int(each[1]) - 1] = 1
	tmp4[int(each[2]) - 1] = 1
	tmp321[int(each[3]) - 1] = 1
	tmp322[int(each[4]) - 1] = 1
	tmp = [each[0], each[5]]
	tmp.extend(tmp5)
	tmp.extend(tmp4)
	tmp.extend(tmp321)
	tmp.extend(tmp322)
	X_data.append(tmp)

X_data = np.array(X_data)
# print X_data[0]

keep_prob = tf.placeholder("float")
w1 = tf.Variable(tf.random_normal([input_num, hid1], stddev=1))
w2 = tf.Variable(tf.random_normal([hid1,output], stddev=1))
b1 = tf.Variable(tf.zeros([1, hid1])+0.1)
b2 = tf.Variable(tf.zeros([1, output])+0.1)
x  = tf.placeholder(tf.float32, shape=(None,input_num), name="input")
a = tf.nn.relu(tf.matmul(x, w1) + b1)
y = tf.nn.relu(tf.matmul(a, w2) + b2)
y_ = tf.placeholder(tf.float32, shape=(None, 1), name = "label")

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
def new_batch(batch_size):
	batch_idx = np.random.choice(range(X_data.shape[0]), size = batch_size, replace = False)
	batch_x = np.zeros((batch_size, input_num))
	batch_y_ = np.zeros((batch_size, 1))
	for i in range(batch_size):
		batch_x[i] = X_data[batch_idx[i]]
		batch_y_[i] = Y_data[batch_idx[i]]
	return batch_x, batch_y_

X_val, y_val = new_batch(1000)
def mape(y_true, y_pred):
	return tf.reduce_mean(tf.reduce_sum(abs(y_true-y_pred)/y_true))

for i in range(iterations):
	batch_x, batch_y_ = new_batch(batch_size)
	sess.run(train_step, feed_dict = {x: batch_x, y_: batch_y_, keep_prob:drop_keep_prob})
	if i % (iterations // 100) == 0:
		print ('Process: {}%'.format((i // (iterations // 100) + 1) * 1))
		print sess.run(y_[0], feed_dict = {x: batch_x, y_: batch_y_, keep_prob:drop_keep_prob})
		print sess.run(y[0], feed_dict = {x: batch_x, y_: batch_y_, keep_prob:drop_keep_prob})
		correct_prediction = mape(y_, y)
		print ('MAPE:', sess.run(correct_prediction, feed_dict = {x: X_val, y_: y_val, keep_prob: 1.0}))
