import tensorflow.compat.v1 as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.disable_v2_behavior()
# 通过操作符号变量来描述这些可交互的操作单元，可以用下面的方式创建一个
# x 不是一个特定的值,而是一个占位符 placeholder，我们在 TensorFlow 运行计算时输入这个值.
# 我们希望能够输入任意数量的 MNIST 图像，每一张图展平成 784 维的向量
x = tf.placeholder("float",[None,784])


# Variable 一个 Variable 代表一个可修改的张量，存在在 TensorFlow 的用于描述交互性操作的图中。
# 它们可以用于计算输入值，也可以在计算中被修改。对于各种机器学习应用，一般都会有模型参数，可以用 Variable 表示。
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# tf.matmul(X,W) 表示 x 乘以 W，对应之前等式里面的，这里 x 是一个 2 维张量拥有多个输入
# tf.matmul(a, b....)将矩阵 a 乘以矩阵 b,生成a * b
y = tf.nn.softmax(tf.matmul(x, W) + b)

# ==================================================
# 训练模型

y_ = tf.placeholder("float", [None, 10])

#  tf.log 计算 y 的每个元素的对数
# 我们把 y_ 的每一个元素和 tf.log(y_) 的对应元素相乘
# 用 tf.reduce_sum 计算张量的所有元素的总和
# tf.reduce_sum(input_tensor,axis=None,...) 此函数计算一个张量的各个维度上元素的总和(可以计算指定维度)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# tf.train 提供了一组帮助训练模型的类和函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys})

# 评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

