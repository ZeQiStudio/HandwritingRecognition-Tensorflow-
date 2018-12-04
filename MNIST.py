import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DACAY = 0.99

# First Version inference()
'''
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
'''


def inference(input_tensor, avg_class, reuse=False):
    if avg_class == None:
        with tf.variable_scope('layer1', reuse=reuse):
            weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(1.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        with tf.variable_scope('layer2', reuse=reuse):
            weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(1.0))
            layer2 = tf.matmul(layer1, weights) + biases
        return layer2
    else:
        with tf.variable_scope('layer1', reuse=reuse):
            weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(1.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))
        with tf.variable_scope('layer2', reuse=reuse):
            weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(1.0))
            layer2 = tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases)
        return layer2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    '''
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    '''

    y = inference(x, None)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DACAY, global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, True)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    with tf.variable_scope('', reuse=True):
        weights1 = tf.get_variable('layer1/weights', [INPUT_NODE, LAYER1_NODE])
        weights2 = tf.get_variable('layer2/weights', [LAYER1_NODE, OUTPUT_NODE])
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess,"./model.ckpt")
        #saver.save(sess, "./model.ckpt")
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            if i % 1000 == 0:
                validation_acc = sess.run(accuracy, feed_dict=validate_feed)
                loss_v = sess.run(loss, feed_dict={x: xs, y_: ys})
                print('After %d steps,validation_accuracy is %g , loss is %g' % (i, validation_acc, loss_v))
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        validate_acc = sess.run(accuracy, feed_dict=validate_feed)
        print('After %d steps,test_accuracy is %g,validation_acc is %g' % (TRAINING_STEPS, test_acc, validate_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
