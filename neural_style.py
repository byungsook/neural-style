import time
import tensorflow as tf
import numpy as np
from vgg import VGG19

def check_time(func):
    def new_func(*args, **kwargs):
        print 'Start: %s' % time.ctime()
        start = time.time()
        result = func(*args, **kwargs)
        print 'End: %s' % time.ctime()
        end = time.time()
        elapsed = end - start
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        print 'Elapsed Time: %d:%d:%d [hms]' % (h, m, s)
        return result
    return new_func

@check_time
def reconstruct_content(content_arr, vgg_path, layer, learning_rate, num_iter):
    """ reconstruct content image """
    sess = tf.InteractiveSession()

    # create VGG19 pretrained instance
    vgg19 = VGG19(vgg_path)

    # compute content features in feed forward mode
    shape = (1,) + content_arr.shape # [1, h, w, c]
    input_image = tf.placeholder(tf.float32, shape=shape)
    content_feature_network = vgg19.network(input_image)
    content_pre = np.array([vgg19.preprocess(content_arr)])
    content_feature_map = content_feature_network[layer].eval(feed_dict={input_image: content_pre})

    # # save feature map 0
    # content_feature_map0 = vgg19.unprocess(content_feature_map[0, :, :, 0:1])
    # return content_feature_map0

    # build a model for the content image reconstruction
    initial_noise_image = tf.random_normal(shape, stddev=0.256)
    reconstructed_image = tf.Variable(initial_noise_image)
    reconstruction_network = vgg19.network(reconstructed_image)

    content_loss = 2.0 / content_feature_map.size * (
        tf.nn.l2_loss(content_feature_map - reconstruction_network[layer]))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(content_loss)

    # train the model
    best_loss = float('inf')
    best_image = None
    sess.run(tf.initialize_all_variables())

    for i in range(num_iter):
        train_step.run()
        if i % 50 == 0 or i == num_iter-1:
            step_loss = content_loss.eval()
            print 'step %d, loss %g, best %g' % (i, step_loss, best_loss)
            if step_loss < best_loss:
                best_loss = step_loss
                best_image = reconstructed_image.eval()

    return vgg19.unprocess(best_image.reshape(shape[1:]))