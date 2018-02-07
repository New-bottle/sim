import tensorflow as tf
import numpy as np
import pickle
import sys
import os

print ('------------------LOAD START---------------------')

# train_session_folder = '../models/'
# train_session_name = '1'
# train_session_path = train_session_folder + train_session_name
# train_graph_path = train_session_folder + train_session_name + "-20.meta"

train_file = '../data/training.pk1'
test_file = '../data/test.pk1'

with open(train_file, 'rb') as f:
	X_tmp = np.array(pickle.load(f))
	Y_data = np.array(pickle.load(f))

def num_to_list(num, low, high):
	new_list = [0 for i in range(high - low + 1)]
	new_list[int(num) - low] = 1
	return new_list

def normalize(data):
	m = data.shape[0]
	n = data.shape[1]
	max_data = [1 for i in range(n)]
	for i in range(m):
		for j in range(n):
			max_data[j] = max(max_data[j], data[i][j])
	print(max_data)
	for i in range(m):
		for j in range(n):
			data[i][j] /= max_data[j]

X_data = []
for each in X_tmp:
	tmp = [each[0], each[5]]
	tmp.extend(num_to_list(each[1], 1, 5))
	tmp.extend(num_to_list(each[2], 1, 4))
	tmp.extend(num_to_list(each[3], 1, 32))
	tmp.extend(num_to_list(each[4], 1, 32))
	X_data.append(tmp)
X_data = np.array(X_data)
normalize(X_data)

Y_data = Y_data.reshape(100000,1)
normalize(Y_data)

with open(test_file, 'rb') as f:
	X_test = np.array(pickle.load(f))

print(X_data.shape)
print(Y_data.shape)

print ('-------------------LOAD END----------------------')

np.set_printoptions(suppress=True)
X_data = X_data[0:10]
Y_data = Y_data[0:10]
print(X_data.shape)
print(X_data)
print(Y_data.shape)
print(Y_data * 1992)

input_num = 2+5+4+32+32
hid1 = 100
# hid2 = 100
output_num = 1

learning_rate = 0.05

# 需要模改
def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases			# biases 和 结果 维数不同?
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x = tf.placeholder(tf.float32, shape=[None,input_num], name="input")
y = tf.placeholder(tf.float32, shape=[None,output_num], name = "label")

l1 = add_layer(x, input_num, hid1, activation_function = tf.nn.sigmoid)
# l2 = add_layer(l1, hid1, hid2, activation_function = tf.nn.sigmoid)
y_pred = add_layer(l1, hid1, output_num, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - y), reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# print(train_session_path)
# print(train_graph_path)

# if os.path.exists(train_graph_path) == True:
# 	loader = tf.train.import_meta_graph(train_graph_path)
# 	loader.restore(sess,tf.train.latest_checkpoint(train_session_folder))

# saver = tf.train.Saver()
# saver.save(sess, train_session_path, global_step=20)

for step in range(20001):
    sess.run(train_step, feed_dict = {x: X_data, y: Y_data})
    if step % 200 == 0:
        print(step, np.array([elem * 1992 for elem in sess.run(y_pred, feed_dict = {x: X_data, y: Y_data})]),  sess.run(loss, feed_dict = {x: X_data, y: Y_data}))
