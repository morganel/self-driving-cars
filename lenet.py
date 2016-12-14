"""
LeNet Architecture

HINTS for layers:

    Convolutional layers:

    tf.nn.conv2d
    tf.nn.max_pool

    For preparing the convolutional layer output for the
    fully connected layers.

    tf.contrib.flatten
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten


EPOCHS = 10
BATCH_SIZE = 50
n_classes = 10

# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the LeNet and return the result of the last fully connected layer.

layer_width = {
    'conv_layer_1': 6,
    'conv_layer_2': 16,
    'fully_connected': 120
}

weight = {
    'conv_layer_1':tf.Variable(tf.truncated_normal([5,5,1,layer_width['conv_layer_1']])),     
    'conv_layer_2':tf.Variable(tf.truncated_normal([5,5,
            layer_width['conv_layer_1'],
            layer_width['conv_layer_2']])),
    'fully_connected': tf.Variable(tf.truncated_normal(
        [5*5*16, layer_width['fully_connected']])),
    'out': tf.Variable(tf.truncated_normal(
        [layer_width['fully_connected'], n_classes])),
    }

bias = {
    'conv_layer_1':tf.Variable(tf.zeros(layer_width['conv_layer_1'])),
    'conv_layer_2':tf.Variable(tf.zeros(layer_width['conv_layer_2'])),
    'fully_connected':tf.Variable(tf.zeros(layer_width['fully_connected'])),
    'out':tf.Variable(tf.zeros(n_classes))
    }

def LeNet(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    x = tf.reshape(x, (-1, 28, 28, 1))
    # Pad 0s to 32x32. Centers the digit further.
    # Add 2 rows/columns on each side for height and width dimensions.
    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")
    # TODO: Define the LeNet architecture.
    print(x)
    # 32x32x1
    #Convolution layer 1. The output shape should be 28x28x6.
    conv_layer_1 = tf.nn.conv2d(x, weight['conv_layer_1'], strides=[1, 1, 1, 1], padding='VALID')
    conv_layer_1 = tf.nn.bias_add(conv_layer_1, bias['conv_layer_1'])
    #Activation 1. Your choice of activation function.
    conv_layer_1 = tf.nn.relu(conv_layer_1)
    
    #Pooling layer 1. The output shape should be 14x14x6.
    layer_1 = tf.nn.max_pool(conv_layer_1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
    
    #Convolution layer 2. The output shape should be 10x10x16.
    conv_layer_2 = tf.nn.conv2d(layer_1, weight['conv_layer_2'], strides=[1, 1, 1, 1], padding='VALID')
    conv_layer_2 = tf.nn.bias_add(conv_layer_2, bias['conv_layer_2'])
    
    #Activation 2. Your choice of activation function.
    conv_layer_2 = tf.nn.relu(conv_layer_2)
    #Pooling layer 2. The output shape should be 5x5x16.
    layer_2 = tf.nn.max_pool(conv_layer_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
    #Flatten layer. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.
    print(layer_2)
    flat_layer = tf.contrib.layers.flatten(layer_2)
    
    #flat_layer = flatten(layer_2)
    # (5 * 5 * 16, 120)
    fc1_shape = (flat_layer.get_shape().as_list()[-1], 120)
    print(flat_layer)
    print(fc1_shape)
    
    #could use fc1_shape in weight['fully_connected']

    #Fully connected layer 1. This should have 120 outputs.
    fully_connected = tf.add(tf.matmul(flat_layer, weight['fully_connected']), bias['fully_connected'])
    print(fully_connected)
    #Activation 3. Your choice of activation function.
    fully_connected = tf.nn.relu(fully_connected)
    #Fully connected layer 2. This should have 10 outputs.
    
    out = tf.add(tf.matmul(fully_connected, weight['out']), bias['out'])
    #You'll return the result of the 2nd fully connected layer from the LeNet function.

    # Return the result of the last fully connected layer.
    return out


# MNIST consists of 28x28x1, grayscale images
x = tf.placeholder(tf.float32, (None, 784))
# Classify over 10 digits 0-9
y = tf.placeholder(tf.float32, (None, 10))
fc2 = LeNet(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def eval_data(dataset):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    # If dataset.num_examples is not divisible by BATCH_SIZE
    # the remainder will be discarded.
    # Ex: If BATCH_SIZE is 64 and training set has 55000 examples
    # steps_per_epoch = 55000 // 64 = 859
    # num_examples = 859 * 64 = 54976
    #
    # So in that case we go over 54976 examples instead of 55000.
    steps_per_epoch = dataset.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    for step in range(steps_per_epoch):
        batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_acc/num_examples


if __name__ == '__main__':
    # Load data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        steps_per_epoch = mnist.train.num_examples // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        # Train model
        for i in range(EPOCHS):
            for step in range(steps_per_epoch):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            val_loss, val_acc = eval_data(mnist.validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation loss = {:.3f}".format(val_loss))
            print("Validation accuracy = {:.3f}".format(val_acc))
            print()

        # Evaluate on the test data
        test_loss, test_acc = eval_data(mnist.test)
        print("Test loss = {:.3f}".format(test_loss))
        print("Test accuracy = {:.3f}".format(test_acc))


