from sklearn.utils import shuffle
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def dense(x, n_labels, p_drop=0.5, mu=0.0, sigma=0.1):
    '''
    Build a three layer dense model with dropout as the first baseline
    '''
    i = tf.compat.v1.layers.flatten(x)
    dense1_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(1024, 512), mean = mu, stddev = sigma))
    dense1_b = tf.Variable(tf.zeros(512))
    dense1   = tf.nn.relu(tf.matmul(i, dense1_W) + dense1_b)
    
    dense2_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(512, 256), mean = mu, stddev = sigma))
    dense2_b = tf.Variable(tf.zeros(256))
    dense2   = tf.nn.relu(tf.matmul(dense1, dense2_W) + dense2_b)
    
    dense3_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(256, 128), mean = mu, stddev = sigma))
    dense3_b = tf.Variable(tf.zeros(128))
    dense3   = tf.nn.relu(tf.matmul(dense2, dense3_W) + dense3_b)    

    dropout  = tf.nn.dropout(dense3, p_drop)
    dense4_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(128, n_labels), mean = mu, stddev = sigma))
    dense4_b = tf.Variable(tf.zeros(n_labels))    
    logits   = tf.matmul(dropout, dense4_W) + dense4_b
    return logits


def lenet(x, n_labels, mu=0.0, sigma=0.1):    
    '''
    Build the lenet convnet from the lecture as the second baseline
    '''
    conv1_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    conv2_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    fc0   = tf.compat.v1.layers.flatten(conv2)
    
    fc1_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1    = tf.nn.relu(fc1)

    fc2_W  = tf.Variable(tf.compat.v1.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2    = tf.nn.relu(fc2)

    fc3_W  = tf.Variable(tf.compat.v1.truncated_normal(shape=(84, n_labels), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_labels))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits


def lenet_pp(x, n_labels, p_drop=0.5, mu=0.0, sigma=0.1):    
    '''
    Tune the lenet architecture.
    I basically added another layer of convolutions and increased the number of kernels for
    all convolutional layers: [32 -> 64 -> 128]. I also added dropout and increased the dense
    layer sizes.
    '''
    # More filters per layer
    conv1_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    conv2_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Add 3x3 convolutional layer
    conv3_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(3, 3, 64, 128), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(128))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    conv3 = tf.nn.relu(conv3)
    
    fc0   = tf.compat.v1.layers.flatten(conv3)
    
    # larger dense layers
    fc1_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(1152, 256), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(256))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1    = tf.nn.relu(fc1)
   
    # Add Dropout
    dropout = tf.nn.dropout(fc1, p_drop)

    fc2_W  = tf.Variable(tf.compat.v1.truncated_normal(shape=(256, 128), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(128))
    fc2    = tf.matmul(dropout, fc2_W) + fc2_b
    fc2    = tf.nn.relu(fc2)
        
    fc3_W  = tf.Variable(tf.compat.v1.truncated_normal(shape=(128, n_labels), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_labels))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits


def train(train_x, train_y, dev_x, dev_y, test_x, test_y, n_labels, model_func, model_name, rate = 0.001, epochs=10, batch=128):
    '''
    Train a neural network and test it's perfomance on the dev set while training
    and on the test set afterwards.
    '''
    x         = tf.compat.v1.placeholder(tf.float32, (None, 32, 32, 1))
    y         = tf.compat.v1.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_labels)
    
    def evaluate(X_data, y_data, acc, batch_size):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.compat.v1.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
            accuracy = sess.run(acc, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    logits             = model_func(x, n_labels)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver              = tf.compat.v1.train.Saver()

    cross_entropy      = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation     = tf.reduce_mean(cross_entropy)
    optimizer          = tf.compat.v1.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        num_examples = len(train_x)

        print("Training: {}".format(model_name))
        print()
        for i in range(epochs):
            train_x, train_y = shuffle(train_x, train_y)
            for offset in range(0, num_examples, batch):
                end = offset + batch
                batch_x, batch_y = train_x[offset:end], train_y[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            validation_accuracy = evaluate(dev_x, dev_y, accuracy_operation, batch)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
        saver.save(sess, 'output/{}'.format(model_name))
        print("Model saved")
        train_accuracy = evaluate(train_x, train_y, accuracy_operation, batch_size=128)
        dev_accuracy = evaluate(dev_x, dev_y, accuracy_operation, batch_size=128)
        test_accuracy = evaluate(test_x, test_y, accuracy_operation, batch_size=128)
        
        print("MODEL: {}".format(model_name))
        print("\tTest Accuracy  = {:.3f}".format(test_accuracy))
        print("\tDev Accuracy   = {:.3f}".format(dev_accuracy))
        print("\tTrain Accuracy = {:.3f}".format(train_accuracy))
        return test_accuracy, dev_accuracy, train_accuracy
