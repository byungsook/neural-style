import tensorflow as tf
from vgg import VGG19
import numpy as np
import time

def check_time(func):
    def new_func(*args, **kwargs):
        print "Start: %s" % time.ctime()
        start = time.time()
        result = func(*args, **kwargs)
        print "End: %s" % time.ctime()
        end = time.time()
        elapsed = end - start
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        print "Elapsed Time: %d:%d:%d [hms]" % (h, m, s)
        return result
    return new_func

@check_time
def reconstruct_content(content_arr, style_arr, vgg_path, layer):
    sess = tf.InteractiveSession()

    # create VGG19 pretrained instance
    vgg19 = VGG19(vgg_path)

    # compute content features in feed forward mode
    content_shape = (1,) + content_arr.shape # [1, h, w, c]
    input_image = tf.placeholder(tf.float32, shape=content_shape)
    network = vgg19.network(input_image)
    content_pre = np.array([vgg19.preprocess(content_arr)])
    content_features = network[layer].eval(feed_dict={input_image: content_pre})

    # save feature map 0
    content_features0 = vgg19.unprocess(content_features[0, :, :, 0:1])
    return content_features0
