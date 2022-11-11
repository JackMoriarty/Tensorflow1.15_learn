from ast import With
from cProfile import label
from importlib_metadata import _top_level_inferred
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
# 获取MNIST 数据集
mnist = input_data.read_data_sets('/tmp/data/mnist', one_hot=True)
# 输入模块
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_innput')
#softmax 网络层
with tf.name_scope('softmax_layer'):
    with tf.name_scope('weights'):
        weights = tf.Variable(tf.zeros([784, 10]))
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.zeros([10]))
    with tf.name_scope('Wx_plus_b'):
        y = tf.matmul(x, weights) + biases
# 交叉熵
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)
# 优化器
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# 准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建交互式回话
sess = tf.InteractiveSession()
# 创建 FileWriter实例，并传入当前会话加载的数据流图
writer = tf.summary.FileWriter('/tmp/summary/mnist', sess.graph)
# 初始化全局变量
tf.global_variables_initializer().run()
# 关闭FileWriter的输出流
writer.close()