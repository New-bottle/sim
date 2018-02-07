import tensorflow as tf
import numpy as np
import pickle
import sys
import os

print ('------------------LOAD START---------------------')

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
X_data = X_data[0:100]
Y_data = Y_data[0:100]
print(X_data.shape)
# print(X_data)
print(Y_data.shape)
# print(Y_data * 1992)

input_num = 2+5+4+32+32
hid1 = 100
# hid2 = 50
output_num = 1

learning_rate = 0.01

# 需要模改
def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases			# biases 和 结果 维数不同?
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x = tf.placeholder(tf.float32, shape=[None,input_num], name="input")
y = tf.placeholder(tf.float32, shape=[None,output_num], name = "label")

l1 = add_layer(x, input_num, hid1, activation_function = tf.nn.sigmoid)
# l2 = add_layer(l1, hid1, hid2, activation_function = tf.nn.relu)
y_pred = add_layer(l1, hid1, output_num, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - y), reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()

def run_dp(origin_step, train_num):
	init = tf.global_variables_initializer()

	sess = tf.Session()
	sess.run(init)

	models_folder = '../models/'
	models_name = '1'
	models_full_name = models_folder + models_name
	models_graph_path = models_folder + models_name + '-' + str(origin_step) + '.meta'

	print('models_graph_path: %s',models_graph_path)
	if os.path.exists(models_graph_path) == True:
		loader = tf.train.import_meta_graph(models_graph_path)
		loader.restore(sess,tf.train.latest_checkpoint(models_folder))
		new_step = origin_step + train_num
	else:
		new_step = train_num

	for step in range(train_num + 1):
		sess.run(train_step, feed_dict = {x: X_data, y: Y_data})
		if step % 1e8 == 0:
			# learning_rate *= 0.95
			# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
			print(new_step - train_num + step, np.array([elem * 1992 for elem in sess.run(y_pred, feed_dict = {x: X_data, y: Y_data})]),  sess.run(loss, feed_dict = {x: X_data, y: Y_data}))

	saver.save(sess, models_full_name, global_step = new_step)

for i in range(1000):
	epoch = 2000
	run_dp(i * epoch, epoch)