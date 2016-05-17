import time
import tensorflow as tf
import numpy as np
import sklearn.decomposition
from vgg import VGG19

try:
    reduce
except NameError:
    from functools import reduce

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

    # # save content feature map 0
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

@check_time
def reconstruct_style(style_arr, vgg_path, layers, learning_rate, num_iter):
    """ reconstruct style image """
    sess = tf.InteractiveSession()

    # create VGG19 pretrained instance
    vgg19 = VGG19(vgg_path)

    # compute style features in feed forward mode
    shape = (1,) + style_arr.shape # [1, h, w, c]
    input_image = tf.placeholder(tf.float32, shape=shape)
    style_feature_network = vgg19.network(input_image)
    style_pre = np.array([vgg19.preprocess(style_arr)])

    style_gram = {}
    for layer in layers:
        style_feature_map = style_feature_network[layer].eval(feed_dict={input_image: style_pre})
        style_feature_map = np.reshape(style_feature_map, (-1, style_feature_map.shape[3]))
        style_gram[layer] = np.matmul(style_feature_map.T, style_feature_map) / style_feature_map.size


    # # save style gram
    # return style_gram

    # build a model for the style image reconstruction
    initial_noise_image = tf.random_normal(shape, stddev=0.256)
    reconstructed_image = tf.Variable(initial_noise_image)
    reconstruction_network = vgg19.network(reconstructed_image)

    style_loss = 0
    for layer in layers:
        style_feature_map = reconstruction_network[layer]
        _, height, width, depth = map(lambda s: s.value, style_feature_map.get_shape())
        style_feature_map = tf.reshape(style_feature_map, (-1, depth))
        style_feature_map_size = height*width*depth
        gram = tf.matmul(style_feature_map, style_feature_map, transpose_a=True) / style_feature_map_size
        style_loss += 2.0 * tf.nn.l2_loss(gram - style_gram[layer]) / style_gram[layer].size

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(style_loss)

    # train the model
    best_loss = float('inf')
    best_image = None
    sess.run(tf.initialize_all_variables())

    for i in range(num_iter):
        train_step.run()
        if i % 50 == 0 or i == num_iter-1:
            step_loss = style_loss.eval()
            print 'step %d, loss %g, best %g' % (i, step_loss, best_loss)
            if step_loss < best_loss:
                best_loss = step_loss
                best_image = reconstructed_image.eval()

    return vgg19.unprocess(best_image.reshape(shape[1:]))


def synthesize_image(content_arr, style_arr, vgg_path,
                     content_layer, style_layers,
                     content_weight, style_weight, tv_weight,
                     learning_rate, num_iter):
    """ synthesize content and style images """
    print 'Start: %s' % time.ctime()
    start = time.time()
        
    sess = tf.InteractiveSession()

    # create VGG19 pretrained instance
    vgg19 = VGG19(vgg_path)

    # compute content features in feed forward mode
    content_shape = (1,) + content_arr.shape # [1, h, w, c]
    input_content_image = tf.placeholder(tf.float32, shape=content_shape)
    content_feature_network = vgg19.network(input_content_image)
    content_pre = np.array([vgg19.preprocess(content_arr)])
    content_feature_map = content_feature_network[content_layer].eval(
        feed_dict={input_content_image: content_pre})

    # compute style features in feed forward mode
    style_shape = (1,) + style_arr.shape # [1, h, w, c]
    input_style_image = tf.placeholder(tf.float32, shape=style_shape)
    style_feature_network = vgg19.network(input_style_image)
    style_pre = np.array([vgg19.preprocess(style_arr)])

    style_gram = {}
    for style_layer in style_layers:
        style_feature_map = style_feature_network[style_layer].eval(
            feed_dict={input_style_image: style_pre})
        style_feature_map = np.reshape(style_feature_map, (-1, style_feature_map.shape[3]))
        style_gram[style_layer] = (np.matmul(style_feature_map.T, style_feature_map) /
                                   style_feature_map.size)


    # build a model for the image synthesis
    initial_noise_image = tf.random_normal(content_shape, stddev=0.256)
    synthesized_image = tf.Variable(initial_noise_image)
    synthesis_network = vgg19.network(synthesized_image)

    # content loss
    content_loss = content_weight * 2.0 / content_feature_map.size * (
        tf.nn.l2_loss(content_feature_map - synthesis_network[content_layer]))

    # sytle loss
    style_loss = 0
    for style_layer in style_layers:
        style_feature_map = synthesis_network[style_layer]
        _, height, width, depth = map(lambda s: s.value, style_feature_map.get_shape())
        style_feature_map = tf.reshape(style_feature_map, (-1, depth))
        style_feature_map_size = height*width*depth
        gram = tf.matmul(style_feature_map, style_feature_map, transpose_a=True) / style_feature_map_size
        style_loss += style_weight * 2.0 * (tf.nn.l2_loss(gram - style_gram[style_layer]) /
                             style_gram[style_layer].size)

    # total variation denoising
    def _tensor_size(tensor):
        from operator import mul
        return reduce(mul, (d.value for d in tensor.get_shape()), 1)

    tv_y_size = _tensor_size(synthesized_image[:, 1:, :, :])
    tv_x_size = _tensor_size(synthesized_image[:, :, 1:, :])
    tv_loss = tv_weight * 2 * (
        (tf.nn.l2_loss(synthesized_image[:, 1:, :, :] -
                       synthesized_image[:, :content_shape[1]-1, :, :]) / tv_y_size) +
        (tf.nn.l2_loss(synthesized_image[:, :, 1:, :] -
                       synthesized_image[:, :, :content_shape[2]-1, :]) / tv_x_size))

    loss = content_loss + style_loss + tv_loss
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # train the model
    best_loss = float('inf')
    best_image = None
    sess.run(tf.initialize_all_variables())

    for i in range(num_iter):
        train_step.run()
        if i % 50 == 0 or i == num_iter-1:
            step_content_loss = content_loss.eval()
            step_style_loss = style_loss.eval()
            step_tv_loss = tv_loss.eval()
            step_loss = step_content_loss + step_style_loss + step_tv_loss
            print 'step %d, content loss %g, style loss %g, tv loss %g, best %g' % (
                i, step_content_loss, step_style_loss, step_tv_loss, best_loss)
            if step_loss < best_loss:
                best_loss = step_loss
                best_image = synthesized_image.eval()
            yield (i, vgg19.unprocess(best_image.reshape(content_shape[1:])))

    print 'End: %s' % time.ctime()
    end = time.time()
    elapsed = end - start
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print 'Elapsed Time: %d:%d:%d [hms]' % (h, m, s)



def filter_style_image(style_arr, vgg_path, layers):
    """ save filtered style image """    
    print 'Start: %s' % time.ctime()
    start = time.time()
    
    sess = tf.InteractiveSession()

    # create VGG19 pretrained instance
    vgg19 = VGG19(vgg_path)

    # compute style features in feed forward mode
    shape = (1,) + style_arr.shape # [1, h, w, c]
    input_image = tf.placeholder(tf.float32, shape=shape)
    style_feature_network = vgg19.network(input_image)
    style_pre = np.array([vgg19.preprocess(style_arr)])

    for i, layer in enumerate(layers):
        style_feature_map = style_feature_network[layer].eval(feed_dict={input_image: style_pre})
        yield (i, style_feature_map)

    print 'End: %s' % time.ctime()
    end = time.time()
    elapsed = end - start
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print 'Elapsed Time: %d:%d:%d [hms]' % (h, m, s)


def svd_gram_style_image(style_arr, vgg_path, layers):
    """ save filtered style image """    
    print 'Start: %s' % time.ctime()
    start = time.time()
    
    sess = tf.InteractiveSession()

    # create VGG19 pretrained instance
    vgg19 = VGG19(vgg_path)

    # compute style features in feed forward mode
    shape = (1,) + style_arr.shape # [1, h, w, c]
    input_image = tf.placeholder(tf.float32, shape=shape)
    style_feature_network = vgg19.network(input_image)
    style_pre = np.array([vgg19.preprocess(style_arr)])

    for i, layer in enumerate(layers):
        style_feature_map = style_feature_network[layer].eval(feed_dict={input_image: style_pre})
        style_feature_map_size = style_feature_map.shape
        style_feature_map = np.reshape(style_feature_map, (-1, style_feature_map.shape[3]))
        # svd
        # U, s, V = np.linalg.svd(style_feature_map, full_matrices=False)
        rank = 16
        tsvd = sklearn.decomposition.TruncatedSVD(rank, algorithm="randomized", n_iter=1)
        x_truncated = tsvd.fit_transform(style_feature_map)
        sorted_id = np.argsort(tsvd.explained_variance_ratio_)[::-1]
        x_truncated = x_truncated[:, sorted_id]
        sorted_var_ratio = tsvd.explained_variance_ratio_[sorted_id]
        print sorted_var_ratio, sum(sorted_var_ratio)
        # x_norm = (x_truncated - x_truncated.min(0)) / x_truncated.ptp(0) * 255.0
        x_norm = x_truncated # without normalization
        yield (i, x_norm.reshape(
            style_feature_map_size[1], style_feature_map_size[2], rank),
            sorted_var_ratio)

    print 'End: %s' % time.ctime()
    end = time.time()
    elapsed = end - start
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print 'Elapsed Time: %d:%d:%d [hms]' % (h, m, s)