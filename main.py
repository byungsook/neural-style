import os

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import neural_style

CONTENT_PATH = 'examples/1-content.jpg'
STYLE_PATH = 'examples/1-style.jpg'
OUTPUT_PATH = 'examples/1-output2.jpg'
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
LEARNING_RATE = 1e1
NUM_ITER = 1000

def main():
    current_path = os.getcwd()
    debug = True
    if not current_path.endswith('neural-style'):
        debug = False
        working_path = current_path + '/neural-style'
        os.chdir(working_path)
    print 'current path', os.getcwd()

    # load content image
    content_arr = imread(CONTENT_PATH)

    if not debug:
        content_arr = content_arr / 255 # normalize
        print content_arr, content_arr.shape
        plt.imshow(content_arr)
        plt.show()

    # # load pretrained network matrix
    # if not os.path.isfile(VGG_PATH):
    #     parser.error("Network %s does not exist. (Did you forget to download it?)" % VGG_PATH)

    # load style image
    # style_images = [imread(style) for style in options.styles]
    style_arr = imread(STYLE_PATH)

    # resize style image
    target_shape = content_arr.shape
    style_scale = float(target_shape[1]) / style_arr.shape[1]
    style_arr = scipy.misc.imresize(style_arr, style_scale) # include normalization

    if not debug:
        print style_arr, style_arr.shape
        plt.imshow(style_arr)
        plt.show()


    # # reconstruct content image
    # rec_content_layer = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'relu4_2', 'conv5_1')
    # for layer in rec_content_layer:
    #     reconstructed_image = neural_style.reconstruct_content(
    #         content_arr, VGG_PATH, layer, LEARNING_RATE, NUM_ITER)
    #     imsave('output/1-rec-content-'+layer+'.jpg', reconstructed_image)

    # layer = 'relu4_2'
    # reconstructed_image = neural_style.reconstruct_content(
    #     content_arr, VGG_PATH, layer, LEARNING_RATE, NUM_ITER)
    # imsave('output/1-rec-content-'+layer+'.jpg', reconstructed_image)

    # reconstruct style image
    rec_style_layer = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
    for i in range(len(rec_style_layer)):
        reconstructed_image = neural_style.reconstruct_style(
            style_arr, VGG_PATH, rec_style_layer[0:i+1], LEARNING_RATE, NUM_ITER)
        imsave('output/1-rec-style-'+rec_style_layer[i]+'.jpg', reconstructed_image)
                

def imread(file_name):
    """ load image and cast type to float """
    return scipy.misc.imread(file_name).astype(np.float)

def imsave(file_name, image):
    """ clip image to range 0-255, cast type to uint8 and save """
    image = np.clip(image, 0, 255).astype(np.uint8)
    scipy.misc.imsave(file_name, image)

if __name__ == '__main__':
    main()
