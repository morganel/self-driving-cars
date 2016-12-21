
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission, if necessary. Sections that begin with **'Implementation'** in the header indicate where you should begin your implementation for your project. Note that some sections of implementation are optional, and will be marked with **'Optional'** in the header.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# 
# ## Step 1: Dataset Exploration
# 
# Visualize the German Traffic Signs Dataset. This is open ended, some suggestions include: plotting traffic signs images, plotting the count of each sign, etc. Be creative!
# 
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - features -> the images pixel values, (width, height, channels)
# - labels -> the label of the traffic sign
# - sizes -> the original width and height of the image, (width, height)
# - coords -> coordinates of a bounding box around the sign in the image, (x1, y1, x2, y2). Based the original image (not the resized version).

# In[1]:

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from sklearn import preprocessing as skp


# In[2]:

# Load pickled data
import pickle

# TODO: fill this in based on where you saved the training and testing data
training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


# ### if we want to downsample to only take a few classes into account

# In[3]:

downsample = False


# In[4]:

def select_classes_only(X, Y, classes):
    ix = np.in1d(Y, classes)
    
    X = X[ix]
    Y = Y[ix]
    
    d = dict(zip(np.unique(Y), range(len(classes))))
    for key,value in d.items():
        Y[Y == key] = value
    #y_train = np.asarray([d[x] for x in y_train])
    return X,Y

if(downsample == True):
    selected_classes = [1,2,4]
    X_train, y_train = select_classes_only(X_train, y_train, selected_classes)
    X_test, y_test = select_classes_only(X_test, y_test, selected_classes)


# In[5]:

### To start off let's do a basic data summary.

# TODO: number of training examples
n_train = len(X_train)

# TODO: number of testing examples
n_test = len(X_test)

# TODO: what's the shape of an image?
image_shape = X_train[0].shape

# TODO: how many classes are in the dataset
n_classes = len(np.unique(np.concatenate((y_train,y_test),axis=0)))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# In[11]:

y = pd.DataFrame(y_train)
y.columns = ['class']
y['index'] = y.index

gb = y.groupby(['class'])
#print(gb['index'].agg([np.min,np.max]))


# ## Validation Set
# Each image here is almost the same (first ) 
# => for each class, take the first 20% of the indices as validation set
# Then, randomize

# In[14]:

validation_percentage = .2

#def get_validaton_per_class(df, c):
y_train_pd = pd.DataFrame(y_train)
y_train_pd.columns = ['class']
y_train_pd['index'] = y_train_pd.index

def get_validation_set(x, y, c):
    class_sample = y[y['class']== c].shape[0]
    
    val_samples = int(validation_percentage * class_sample)
    train_samples = class_sample - val_samples
    
    yf_val = y[y['class']== c].head(val_samples)#.sample(num_image)
    idx_val = (yf_val['index'])
    
    yf_train = y[y['class']== c].tail(train_samples)#.sample(num_image)
    idx_train = (yf_train['index'])

    return idx_val, idx_train

for cl in range(n_classes):

    idx_val_temp, idx_train_temp = get_validation_set(X_train, y_train_pd, cl)
    
    if(cl == 0):
        idx_val = idx_val_temp
        idx_train = idx_train_temp
    else:
        idx_val = np.concatenate([idx_val,idx_val_temp])
        idx_train = np.concatenate([idx_train, idx_train_temp])

X_valid = X_train[idx_val]
y_valid = y_train[idx_val]
X_train = X_train[idx_train]
y_train = y_train[idx_train]

# TODO: INSTEAD OF TAKING THE FIRST 20%, TAKE THE FIRST MULTIPLES OF 30
#print(X_train.shape, idx_val.shape, idx_train.shape)


# ### Randomize data

# In[15]:

def shuffle_together(x,y):
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return x[p],y[p]

X_train, y_train = shuffle_together(X_train, y_train)
X_valid, y_valid = shuffle_together(X_valid, y_valid)


# In[16]:

print('Train:', X_train.shape, y_train.shape, ' Test:', y_valid.shape, X_valid.shape)


# ### Data exploration visualization

# ### number of example from each class
# same distribution of classes for train and test

# In[18]:

unique, counts = np.unique(y_train, return_counts=True)
unique_valid, counts_valid = np.unique(y_valid, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)



# ### Plot examples of traffic signs for each class

# In[19]:

y = pd.DataFrame(y_train)
y.columns = ['class']
y['index'] = y.index

gb = y.groupby(['class'])
#print(gb['index'].agg([np.min,np.max]))

# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Your model can be derived from a deep feedforward net or a deep convolutional network.
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Implementation
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

# ### Preprocess the data here.

# In[20]:

num_channels = 1
image_size = X_train.shape[1]


# ### Convert to gray scale

# In[21]:

def convert_to_gray_scale(X, b = [0.299,0.587,0.114]):
    return np.inner(X, b)

# plot in color


X_train = convert_to_gray_scale(X_train)
X_valid = convert_to_gray_scale(X_valid)
X_test = convert_to_gray_scale(X_test)

# plot same image in gray scale


# ### Reformat and one-hot encoding

# In[22]:

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)

  labels = (np.arange(n_classes) == labels[:,None]).astype(np.float32)
  return dataset, labels

X_train, y_train = reformat(X_train, y_train)
X_valid, y_valid = reformat(X_valid, y_valid)
X_test, y_test = reformat(X_test, y_test)

print('Features set', X_train.shape, X_valid.shape, X_test.shape)
print('Labels set', y_train.shape, y_valid.shape, y_test.shape)


# ### Normalize pixel values between 0 and 1

# In[23]:

#X_train[0]


# In[24]:

def pixel_normalization(dataset):
    return dataset/255 - 0.5

X_train = pixel_normalization(X_train)
X_valid = pixel_normalization(X_valid)
X_test = pixel_normalization(X_test)


# In[25]:

X_valid.shape


# ### functions to help run graph

# In[26]:

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    output_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
        
    return output_batches


# In[27]:

def accuracy_func(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# ### If we want to downsample even more to see if we can easily overfit the data

# In[28]:

print(X_train.shape)
X_train_subset = X_train#[0:20,:,:,:]
y_train_subset = y_train#[0:20,:]
print(X_train_subset.shape)
print(y_train_subset.shape)

y_train_subset[0]
n_values = np.argmax(y_train, axis=1)
print(max(n_values))


# In[29]:

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


# In[30]:

#print(np.unique(np.argmax(y_valid_subset, axis=1)))
print(np.unique(np.argmax(y_train_subset, axis=1)))
#print(np.unique(np.argmax(y_test_subset, axis=1)))
#print(y_train_subset)
print(np.argmax(y_train_subset, axis=1))


# ### Define model

# In[36]:

std = 0.1

graph = tf.Graph()

tf.reset_default_graph()

with graph.as_default():
    
    # Create basic cnn
    # x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")

    tf_train_dataset = tf.placeholder(tf.float32, 
                          shape=[None, image_size, image_size, num_channels])
    tf_train_labels = tf.placeholder(tf.float32, 
                          shape=[None, n_classes])

    #n_val = tf.placeholder(tf.float32, 
    #                      shape=[None, ])
    
    tf_train_all_dataset = tf.constant(X_train_subset)
    tf_valid_dataset = tf.constant(X_valid)
    tf_test_dataset = tf.constant(X_test)
    
    tf_beta = tf.placeholder(tf.float32)
    tf_keep_prob = tf.placeholder(tf.float32)
    tf_learning_rate = tf.placeholder(tf.float32)
      
    layer_width = {
        'conv_layer_1': 32,
        'conv_layer_2': 54,
        'fully_connected_1': 1024,
        'fully_connected_2': 256
    }

    weights = {'conv_layer_1': tf.Variable(tf.truncated_normal([5, 5, num_channels, layer_width['conv_layer_1']], stddev = std), name = 'conv_layer_1_weight'),
              'conv_layer_2': tf.Variable(tf.truncated_normal([5, 5, layer_width['conv_layer_1'], layer_width['conv_layer_2']], stddev = std), name = 'conv_layer_2_weight'),
              'fully_connected_1' : tf.Variable(tf.truncated_normal([486,1024], stddev = 0.1), name = 'fc_1_weight'),
              'fully_connected_2' : tf.Variable(tf.truncated_normal([1024,256], stddev = 0.1), name = 'fc_2_weight'),
              'out': tf.Variable(tf.truncated_normal([layer_width['fully_connected_2'], n_classes], stddev = std), name = 'out_weight'),
              }
    
    biases = {'conv_layer_1': tf.Variable(tf.zeros(layer_width['conv_layer_1']), name = 'conv_layer_1_bias'),
              'conv_layer_2': tf.Variable(tf.zeros(layer_width['conv_layer_2']), name = 'conv_layer_2_bias'),
              'fully_connected_1': tf.Variable(tf.zeros(layer_width['fully_connected_1']), name = 'fc_1_bias'),
              'fully_connected_2': tf.Variable(tf.zeros(layer_width['fully_connected_2']), name = 'fc_2_bias'),
              'out': tf.Variable(tf.zeros(n_classes), name = 'out_bias')
              }
    
    #def model(data1, keep_prob):

    def perc_zero_activation(l):
        s = tf.reduce_sum(l, 0)
        s1 = tf.equal(s, 0)
        s2 = tf.reduce_mean(tf.cast(s1, tf.float32))
        return s2
    
    #Convolution layer 1. The output shape should be 28x28x6.
    with tf.name_scope("First_conv_layer"):
        with tf.name_scope('Conv'):
            conv_layer_1 = tf.nn.conv2d(tf_train_dataset, weights['conv_layer_1'], strides = [1,1,1,1], padding = 'VALID', name="Conv1")
            conv_layer_1 = conv_layer_1 + biases['conv_layer_1']
            tf.histogram_summary('First_conv_layer' + '/pre_activations', conv_layer_1)
        conv_layer_1 = tf.nn.relu(conv_layer_1, name="Relu1")
        tf.histogram_summary('First_conv_layer' + '/activations', conv_layer_1)
     
        zero_activations = perc_zero_activation(conv_layer_1)
        tf.scalar_summary('cv1_zero_activations', zero_activations)
        
        conv_layer_1 = tf.nn.dropout(conv_layer_1, tf_keep_prob)
        #Pooling layer 1. The output shape should be 14x14x6.
    
    with tf.name_scope("example_image") as scope:
      x_image = tf.reshape(tf_train_dataset, [-1,image_size,image_size,1])
      tf.image_summary("Example_images", x_image)
    
    ## Prepare for visualization
    # Take only convolutions of first image, discard convolutions for other images.
    conv1_size = 28
    V = tf.slice(conv_layer_1, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input') #(28, 28, 32)
    V = tf.reshape(V, (conv1_size, conv1_size, layer_width['conv_layer_1']))
    # Reorder so the channels are in the first dimension, x and y follow.
    V = tf.transpose(V, (2, 0, 1))
    # Bring into shape expected by image_summary
    V = tf.reshape(V, (-1, conv1_size, conv1_size, 1)) #32, 28, 28, 1 => only 1 
    tf.image_summary("first_conv", V, max_images = 3)  
    
    V1 = tf.slice(conv_layer_1, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input') #(28, 28, 32)
    V1 = tf.reshape(V, (conv1_size, conv1_size, 1, layer_width['conv_layer_1']))
    #(28, 28, 1, 32)
    # In this section, we visualize the filters of the first convolutional layers
    # We concatenate the filters into one image
    # Credits for the inspiration go to Martin Gorner
    V1_a = V1                 # [5, 5, 1, 32] convlayer_1: #?, 28, 28, 32
    V1pad= tf.zeros([28, 28, 1, 1])        # [5, 5, 1, 4]  - four zero kernels for padding
    # We have a 6 by 6 grid of kernepl visualizations. yet we only have 32 filters
    # Therefore, we concatenate 4 empty filters
    V1_b = tf.concat(3, [V1_a, V1pad, V1pad, V1pad, V1pad])   # [5, 5, 1, 36] => [28, 28, 1, 36]  
    V1_c = tf.split(3, 36, V1_b)         # 36 x [5, 5, 1, 1] # 36 x [28, 28, 1, 1]
    V1_row0 = tf.concat(0, V1_c[0:6])    # [30, 5, 1, 1] # [168, 28, 1, 1]
    V1_row1 = tf.concat(0, V1_c[6:12])   # [30, 5, 1, 1]
    V1_row2 = tf.concat(0, V1_c[12:18])  # [30, 5, 1, 1]
    V1_row3 = tf.concat(0, V1_c[18:24])  # [30, 5, 1, 1]
    V1_row4 = tf.concat(0, V1_c[24:30])  # [30, 5, 1, 1]
    V1_row5 = tf.concat(0, V1_c[30:36])  # [30, 5, 1, 1]
    V1_d = tf.concat(1, [V1_row0, V1_row1, V1_row2, V1_row3, V1_row4, V1_row5]) # [30, 30, 1, 1]
    V1_e = tf.reshape(V1_d, [1, 168, 168, 1])
    tf.image_summary("Visualize_first_activation_layer_all", V1_e)
    
    # In this section, we visualize the filters of the first convolutional layers
    # We concatenate the filters into one image
    # Credits for the inspiration go to Martin Gorner
    W1_a = weights['conv_layer_1']                    # [5, 5, 1, 32] convlayer_1: #?, 28, 28, 32
    W1pad= tf.zeros([5, 5, 1, 1])        # [5, 5, 1, 4]  - four zero kernels for padding
    # We have a 6 by 6 grid of kernepl visualizations. yet we only have 32 filters
    # Therefore, we concatenate 4 empty filters
    W1_b = tf.concat(3, [W1_a, W1pad, W1pad, W1pad, W1pad])   # [5, 5, 1, 36]  
    W1_c = tf.split(3, 36, W1_b)         # 36 x [5, 5, 1, 1]
    W1_row0 = tf.concat(0, W1_c[0:6])    # [30, 5, 1, 1]
    W1_row1 = tf.concat(0, W1_c[6:12])   # [30, 5, 1, 1]
    W1_row2 = tf.concat(0, W1_c[12:18])  # [30, 5, 1, 1]
    W1_row3 = tf.concat(0, W1_c[18:24])  # [30, 5, 1, 1]
    W1_row4 = tf.concat(0, W1_c[24:30])  # [30, 5, 1, 1]
    W1_row5 = tf.concat(0, W1_c[30:36])  # [30, 5, 1, 1]
    W1_d = tf.concat(1, [W1_row0, W1_row1, W1_row2, W1_row3, W1_row4, W1_row5]) # [30, 30, 1, 1]
    W1_e = tf.reshape(W1_d, [1, 30, 30, 1])
    tf.image_summary("Visualize_weights", W1_e)
    
    with tf.name_scope("Max_pool_layer_1"):
        layer_1 = tf.nn.max_pool(conv_layer_1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID', name="MaxPool1")
        tf.histogram_summary('Max_pool_layer_1' + '/activations', layer_1)
        
    with tf.name_scope("Second_conv_layer"):
        #Convolution layer 2. The output shape should be 10x10x16.
        with tf.name_scope('Conv'):
            conv_layer_2 = tf.nn.conv2d(layer_1, weights['conv_layer_2'], strides = [1,2,2,1], padding = 'SAME', name="Conv2")
            conv_layer_2 = conv_layer_2 + biases['conv_layer_2']
            tf.histogram_summary('Second_conv_layer' + '/pre_activations', conv_layer_2)
        conv_layer_2 = tf.nn.relu(conv_layer_2, name="Relu2")
        tf.histogram_summary('Second_conv_layer' + '/activations', conv_layer_2)
        
        zero_activations_2 = perc_zero_activation(conv_layer_2)
        tf.scalar_summary('cv2_zero_activations', zero_activations_2)
        
        conv_layer_2 = tf.nn.dropout(conv_layer_2, tf_keep_prob)
        #Pooling layer 2. The output shape should be 5x5x16.
    
    ## Prepare for visualization
    # Take only convolutions of first image, discard convolutions for other images.
    conv2_size = 7
    V2 = tf.slice(conv_layer_2, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_second_input') #(28, 28, 32)
    V2 = tf.reshape(V2, (conv2_size, conv2_size, layer_width['conv_layer_2']))
    # Reorder so the channels are in the first dimension, x and y follow.
    V2 = tf.transpose(V2, (2, 0, 1))
    # Bring into shape expected by image_summary
    V2 = tf.reshape(V2, (-1, conv2_size, conv2_size, 1)) #32, 28, 28, 1 => only 1 
    tf.image_summary("second_conv", V2, max_images = 3)  
    
    with tf.name_scope("Max_pool_layer_2"):
        layer_2 = tf.nn.max_pool(conv_layer_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID', name="MaxPool2")
        tf.histogram_summary('Max_pool_layer_2' + '/activations', layer_2)

    # Fully connected
    with tf.name_scope("Flatten"):
        fc = tf.contrib.layers.flatten(layer_2) # (5 * 5 * 16, 120)
        fc_shape = (fc.get_shape().as_list()[-1], layer_width['fully_connected_1'])

    with tf.name_scope("Fully_Connected_layer_1"):
        with tf.name_scope("Wx_plus_b"):
            fc1 = tf.add(tf.matmul(fc, weights['fully_connected_1'], name="FC1_mult"), biases['fully_connected_1'], name = "FC1Sum")
            tf.histogram_summary('Fully_Connected_layer_1' + '/pre_activations', fc1)
        fc1 = tf.nn.relu(fc1, name = "reluFC1")
        tf.histogram_summary('Fully_Connected_layer_1' + '/activations', fc1)
        
        zero_activations_fc1 = perc_zero_activation(fc1)
        tf.scalar_summary('fc1_zero_activations', zero_activations_fc1)
        
        fc1 = tf.nn.dropout(fc1, tf_keep_prob)
        
    with tf.name_scope("Fully_Connected_layer_2"):
        with tf.name_scope("Wx_plus_b"):
            fc2 = tf.add(tf.matmul(fc1, weights['fully_connected_2'], name="FC2_mult"), biases['fully_connected_2'], name = "FC2Sum")
            tf.histogram_summary('Fully_Connected_layer_2' + '/pre_activations', fc2)
        fc2 = tf.nn.relu(fc2, name = "reluFC2")
        tf.histogram_summary('Fully_Connected_layer_2' + '/activations', fc2)
        zero_activations_fc2 = perc_zero_activation(fc2)
        tf.scalar_summary('fc2_zero_activations', zero_activations_fc2)
        
        fc2 = tf.nn.dropout(fc2, tf_keep_prob)
        
    # Out
    with tf.name_scope("Out"):
        with tf.name_scope("Wx_plus_b"):
            logits = tf.add(tf.matmul(fc2, weights['out'], name="out_mult"), biases['out'], name = "out_sum")
            tf.histogram_summary('logits' + '/pre_activations', logits)
        train_prediction = tf.nn.softmax(logits) 
        tf.histogram_summary('logits' + '/activations', train_prediction)
        
    # Optimization
    with tf.name_scope('loss'):

        l2_loss_all = tf.add(tf_beta * 1/2 * tf.nn.l2_loss(weights['conv_layer_1']),
                             tf_beta * 1/2 * tf.nn.l2_loss(weights['conv_layer_2']))
        
        l2_loss_all = tf.add(l2_loss_all, tf_beta * 1/2 * tf.nn.l2_loss(weights['fully_connected_1']))
        l2_loss_all = tf.add(l2_loss_all, tf_beta * 1/2 * tf.nn.l2_loss(weights['fully_connected_2']))
        l2_loss_all = tf.add(l2_loss_all, tf_beta * 1/2 * tf.nn.l2_loss(weights['out']))
            
        data_loss_all = tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
        
        l2_loss = tf.reduce_mean(l2_loss_all)
        data_loss = tf.reduce_mean(data_loss_all)
        
        loss = tf.reduce_mean(l2_loss_all
                            + data_loss_all
                             )    
        
        tf.scalar_summary('l2_loss', l2_loss)
        tf.scalar_summary('data_loss', data_loss) #to check if magnitude of l2_loss is ok relative to data_loss
        tf.scalar_summary('loss', loss)

    with tf.name_scope('SGD'):
        optimizer = tf.train.AdamOptimizer(learning_rate=tf_learning_rate, epsilon = 0.1).minimize(loss) #AdamOptimizer #GradientDescentOptimizer
        
        # Op to calculate every variable gradient
        grads = tf.gradients(loss, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))
            
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
          correct_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(tf_train_labels, 1))
        with tf.name_scope('accuracy'):
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
          tf.scalar_summary('accuracy', accuracy)
    
    tf_learning_rate_array = tf.identity(tf_learning_rate)
    tf_learning_rate_array.set_shape([1])
        
    def get_euclidian_norm(t):
        t_shape = t.get_shape().as_list()   
        dimgrad = np.prod(t_shape[0:])  
        grad2 = tf.reshape(t, [dimgrad]) 
        grad_euclidian_norm = tf.sqrt(tf.reduce_sum(tf.square(grad2)))
        return(grad_euclidian_norm)
    
    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        #print(var.name)
        tf.histogram_summary(var.name, var)
    # Summarize all gradients
    for grad, var in grads:
        tf.histogram_summary(var.name + '/gradient', grad)

        update = tf.mul(grad, -tf_learning_rate_array) 
        update_scale = get_euclidian_norm(update)
        param_scale = get_euclidian_norm(var) #weights['conv_layer_1']
        update_ratio = update_scale / param_scale
        tf.scalar_summary(var.name + '/update_ratio', update_ratio)
        
    merged = tf.merge_all_summaries()
    


# ### Run it:

# In[ ]:

epochs = 1
batch_size = 64
beta = 0.000005 # 0.005
learning_rate = 0.05 #0.005 # 0.05
reload_model = False

timestr = time.strftime("%Y%m%d-%H%M%S")
print(timestr)

save_file = 'model.ckpt'

# Remove the previous weights and bias
tf.reset_default_graph()

with tf.Session(graph = graph) as session:
    saver = tf.train.Saver()
    
    if(reload_model):
        # Load the weights and bias
        saver.restore(session, save_file)
    else:
        session.run(tf.initialize_all_variables()) #tf.initialize_all_variables().run()
    
    train_writer = tf.train.SummaryWriter('log-directory/' + timestr + '/train',session.graph)
    valid_writer = tf.train.SummaryWriter('log-directory/' + timestr + '/valid')
    
    step = 0
    
    for epoch in range(epochs):
        
        if(epoch > 0 and epoch % 3 == 0):
            learning_rate = learning_rate/2
        print('Epoch: ', epoch, ' / learning_rate:', learning_rate)
        for batch_features, batch_labels in batches(batch_size, X_train_subset, y_train_subset):
            step += 1
            
            feed_dict = {tf_train_dataset : batch_features, tf_train_labels : batch_labels, 
                         tf_beta: beta, tf_keep_prob : 0.5, tf_learning_rate : learning_rate}
            
            if step % 100 == 99:  # Record summaries and test-set accuracy

                feed_dict_valid = {tf_train_dataset : X_valid, tf_train_labels : y_valid, 
                                   tf_beta: beta, tf_keep_prob : 1.0, tf_learning_rate : learning_rate}
                summary, acc, l, g = session.run([merged, accuracy, loss, grads], feed_dict=feed_dict_valid)
                valid_writer.add_summary(summary, step)
                print('Validation accuracy at step %s: %s' % (step, acc))
                
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, summary, l, g = session.run([optimizer, merged, loss, grads],
                                      feed_dict=feed_dict,
                                      options=run_options,
                                      run_metadata=run_metadata
                                              )
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                train_writer.add_summary(summary, step)
                print('Adding run metadata for', step)
            
            else:
                
                summary, _ = session.run([optimizer, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                    
    train_writer.close()   
    valid_writer.close()   
    
    # Save the model
    saver.save(session, save_file)
    
    # Test accuracy
    feed_dict_test = {tf_train_dataset : X_test, tf_train_labels : y_test, 
                                   tf_beta: beta, tf_keep_prob : 1.0, tf_learning_rate : learning_rate}
    summary, acc, l = session.run([merged, accuracy, loss], feed_dict=feed_dict_test)
    print('Test accuracy: %s' % (acc))


# In[ ]:

#get_ipython().system('tensorboard --logdir=log-directory/20161220-190551')
#20161220-093624 learning rate 1 and we can see that most activations are 0
#20161220-093949 learning rate 0.05 and we can see that most activations are not 0
#20161220-121831 learning rate 0.05 (good 98.8% on validation)
#20161220-130416 learning rate 0.005 (but then I used a bigger learning rate again when restarting)
# 0.005 really does not work very well even with dropout
# Trying with 0.05 for more epochs


# In[ ]:




# In[ ]:




# In[ ]:

# 3. Are there any duplicates? Like all black pictures / all whites ?


# In[ ]:

# Additional data by rotating image?


# In[ ]:

# Number of examples per label 


# In[ ]:

# Dropout / mini-batch / 


# In[ ]:

# 3. What metric to use: logloss


# ### Question 1 
# 
# _Describe the techniques used to preprocess the data._

# **Answer:**

# In[ ]:

### Generate data additional (if you want to!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.


# ### Question 2
# 
# _Describe how you set up the training, validation and testing data for your model. If you generated additional data, why?_

# **Answer:**

# In[ ]:

### Define your architecture here.
### Feel free to use as many code cells as needed.


# ### Question 3
# 
# _What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)  For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
# ](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._
# 

# **Answer:**

# In[ ]:

### Train your model here.
### Feel free to use as many code cells as needed.


# ### Question 4
# 
# _How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_
# 

# **Answer:**

# ### Question 5
# 
# 
# _What approach did you take in coming up with a solution to this problem?_

# **Answer:**

# ---
# 
# ## Step 3: Test a Model on New Images
# 
# Take several pictures of traffic signs that you find on the web or around you (at least five), and run them through your classifier on your computer to produce example results. The classifier might not recognize some local signs but it could prove interesting nonetheless.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Implementation
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

# In[ ]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.


# ### Question 6
# 
# _Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It would be helpful to plot the images in the notebook._
# 
# 

# **Answer:**

# In[ ]:

### Run the predictions here.
### Feel free to use as many code cells as needed.


# ### Question 7
# 
# _Is your model able to perform equally well on captured pictures when compared to testing on the dataset?_
# 

# **Answer:**

# In[ ]:

### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.


# ### Question 8
# 
# *Use the model's softmax probabilities to visualize the **certainty** of its predictions, [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#top_k) could prove helpful here. Which predictions is the model certain of? Uncertain? If the model was incorrect in its initial prediction, does the correct prediction appear in the top k? (k should be 5 at most)*
# 

# **Answer:**

# ### Question 9
# _If necessary, provide documentation for how an interface was built for your model to load and classify newly-acquired images._
# 

# **Answer:**

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# In[ ]:



