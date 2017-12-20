import pandas as pd 
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm


import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools

## preprocess data
data = pd.read_csv("fer2013/fer2013.csv")

train_data = data[data.Usage == "Training"]
print(train_data.shape)
pixels_values = train_data.pixels.str.split(" ").tolist()
pixels_values = pd.DataFrame(pixels_values, dtype=int)
images = pixels_values.values
images = images.astype(np.float)


images = images - images.mean(axis=1).reshape(-1,1)
images = np.multiply(images,100.0/255.0)


mean_pixel = images.mean(axis=0)
# print("image pix mean shape %d", mean_pixel.shape)
std_pixel = np.std(images, axis=0)

images = np.divide(np.subtract(images,mean_pixel), std_pixel)

num_pixels = images.shape[1]

print("image pixels", num_pixels)
image_width = image_height = np.ceil(np.sqrt(num_pixels)).astype(np.uint8)
labels_num = train_data["emotion"].values
print(labels_num)

labels_count = np.unique(labels_num).shape[0]

def emotion_transfer(data, num_class):
    ans = np.zeros((data.shape[0], num_class))
    for i in range(data.shape[0]):
        ans[i][data[i]] = 1
    return ans

emotions = emotion_transfer(labels_num, labels_count)
emotions = emotions.astype(int)
num_class = 7

validation_size = 1000
training_data = images[validation_size:]
training_label = emotions[validation_size:]

validation_data = images[:validation_size]
validation_label = emotions[:validation_size]

print("training_data", training_data.shape)

## build the cnn model

def weight_init(shape):
    weight = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(weight)

def bias_init(shape):
    bias = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(bias)

def conv(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# input and prediction

image = tf.placeholder('float', shape = [None, num_pixels])
emotion = tf.placeholder('float', shape = [None, num_class])

weight_conv1 = weight_init([5, 5, 1, 50])
bias_conv1 = bias_init([50])

x = tf.reshape(image, [-1, image_height, image_width, 1]) 

conv1 = tf.nn.relu(conv(x, weight_conv1) + bias_conv1)
pool1 = max_pool(conv1)

weight_conv2 = weight_init([5, 5, 50, 100])
bias_conv2 = bias_init([100])

conv2 = tf.nn.relu(conv(pool1, weight_conv2) + bias_conv2)
pool2 = max_pool(conv2)
norm2 = tf.nn.lrn(pool2, 4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75)

weight_full1 = weight_init([100 * 12 * 12, 3000])
bias_full1 = bias_init([3000])

norm2_flat = tf.reshape(norm2, [-1, 12 * 12 * 100])
out1 = tf.nn.relu(tf.matmul(norm2_flat, weight_full1) + bias_full1)

weight_full2 = weight_init([3000, 1000])
bias_full2 = bias_init([1000])

out2 = tf.nn.relu(tf.matmul(out1, weight_full2) + bias_full2)

# dropout to deal with overfit
keep_prob = tf.placeholder('float')
out2_drop = tf.nn.dropout(out2, keep_prob)

weight_full3 = weight_init([1000, num_class])
bias_full3 = bias_init([num_class])

out3 = tf.matmul(out2_drop, weight_full3) + bias_full3

learning_rate = 0.0005

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=emotion, logits=out3))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# compare the pred with the real emotion
correct_prediction = tf.equal(tf.argmax(out3, 1), tf.argmax(emotion, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

#predict the emotion with highest prob
prediction_result = tf.argmax(out3, 1)

iteration = 3000
keep = 0.6
batch_size = 50

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train_accuracies = []
validation_accuracies = []
xlim = []


epochs_completed = 0
index_in_epoch = 0
num_examples = training_data.shape[0]

# serve data by batches
def next_batch(batch_size):
    
    global training_data
    global training_label
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        training_data = training_data[perm]
        training_label = training_label[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return training_data[start:end], training_label[start:end]


for i in range(iteration):
    batch_image, batch_label = next_batch(batch_size)
    # print("batch_image shape is ", batch_image.shape, batch_label.shape)
    if i % 10 == 0 or (i + 1) == iteration:
        train_accuracy = accuracy.eval(feed_dict = {image : batch_image,
                                                    emotion : batch_label,
                                                    keep_prob : 1.})
        validation_accuracy = accuracy.eval(feed_dict = {image : validation_data,
                                                    emotion : validation_label,
                                                    keep_prob : 1.})
        print("Step %d: Training accuracy is %.3f. Validation accuracy is %.3f"%(i, train_accuracy, validation_accuracy))
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        xlim.append(i)
    sess.run(train_step, feed_dict = {image : batch_image, emotion : batch_label, keep_prob : keep})


validation_accuracy = accuracy.eval(feed_dict={ image : validation_data, 
                                                emotion : validation_label, 
                                                keep_prob: 1.0})
print('validation_accuracy => %.3f'%validation_accuracy)
plt.plot(xlim, train_accuracies,'-b', label='Training')
plt.plot(xlim, validation_accuracies,'-g', label='Validation')
plt.legend(loc='lower right', frameon=False)
plt.ylim(ymax = 1.0, ymin = 0.0)
plt.ylabel('accuracy')
plt.xlabel('step')
plt.show()


saver = tf.train.Saver(tf.global_variables())
saver.save(sess, 'my-model1', global_step=0)


test = data[data.Usage == "PublicTest"]


test_pixels_values = test.pixels.str.split(" ").tolist()
test_pixels_values = pd.DataFrame(test_pixels_values, dtype=int)
test_data = test_pixels_values.values
test_data = test_data.astype(np.float)
test_data = test_data - test_data.mean(axis=1).reshape(-1,1)
test_data = np.multiply(test_data,100.0/255.0)
test_data = np.divide(np.subtract(test_data,mean_pixel), std_pixel)

predicted_lables = np.zeros(test_data.shape[0])
for i in range(0,test_data.shape[0]//batch_size):
    predicted_lables[i*batch_size : (i+1)*batch_size] = prediction_result.eval(feed_dict={image : test_data[i*batch_size : (i+1)*batch_size], 
                                                                                keep_prob: 1.0})


print(accuracy_score(test.emotion.values, predicted_lables))
print(confusion_matrix(test.emotion.values, predicted_lables))



cnf_matrix = confusion_matrix(test.emotion.values, predicted_lables)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]*100)/100.0,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title='Confusion Matrix for Test Dataset')

plt.show()
