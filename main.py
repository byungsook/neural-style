import os

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image

import neural_style

CONTENT_PATH = 'examples/1-content.jpg'
STYLE_PATH = 'examples/style.jpg'
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
LEARNING_RATE = 1e1
NUM_ITER = 1000
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2

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

    # layer = 'conv5_1'
    # reconstructed_image = neural_style.reconstruct_content(
    #     content_arr, VGG_PATH, layer, LEARNING_RATE, NUM_ITER)
    # imsave('output/1-rec-content-'+layer+'.jpg', reconstructed_image)

    # # reconstruct style image
    # rec_style_layer = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
    # for i in range(len(rec_style_layer)):
    #     reconstructed_image = neural_style.reconstruct_style(
    #         style_arr, VGG_PATH, rec_style_layer[0:i+1], LEARNING_RATE, NUM_ITER)
    #     imsave('output/1-rec-style-'+rec_style_layer[i]+'.jpg', reconstructed_image)

    # # synthesize image
    # content_layer = ('relu4_2')
    # style_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
    # for i, synthesized_image in neural_style.synthesize_image(
    #     content_arr, style_arr, VGG_PATH,
    #     content_layer, style_layers,
    #     CONTENT_WEIGHT, STYLE_WEIGHT, TV_WEIGHT,
    #     LEARNING_RATE, NUM_ITER):
    #     imsave('output/output-%d.jpg' % i, synthesized_image)

    # # save filtered style image
    # rec_style_layer = ('conv1_1', 'relu1_1', 'conv2_1', 'relu2_1', 'conv3_1', 'relu3_1',
    #     'conv4_1', 'relu4_1', 'conv5_1', 'relu5_1')
    # for i, filtered_style_images in neural_style.filter_style_image(
    #     style_arr, VGG_PATH, rec_style_layer):
    #     depth = filtered_style_images.shape[3]
    #     for j in range(depth):
    #         img = filtered_style_images[0,:,:,j]
    #         imsave('output/filter/filtered-style-%s-%d.jpg' % (rec_style_layer[i], j), img)

    # do SVD for filtered style image
    rec_style_layer = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
    combine_scale = 4
    grid_scale = combine_scale/4.0
    combined_image_size = (int(combine_scale*style_arr.shape[1]), int(combine_scale*style_arr.shape[0])) # w, h
    grid_image_size = (int(grid_scale*style_arr.shape[1]), int(grid_scale*style_arr.shape[0])) # w, h
    for i, reduced_feature_maps, var_ratio in neural_style.svd_gram_style_image(
            style_arr, VGG_PATH, rec_style_layer):
        depth = reduced_feature_maps.shape[2]
        new_img = Image.new('RGB', combined_image_size)
        root_depth = int(depth**0.5)
        j = 0
        for y in range(root_depth):
            for x in range(root_depth):
                img = Image.fromarray(reduced_feature_maps[:, :, j])
                j = j+1
                img = img.resize(grid_image_size, Image.ANTIALIAS)
                new_img.paste(img, (x*grid_image_size[0], y*grid_image_size[1]))
        new_img.save('output/grid/no-norm-grid-style-%s.jpg' % rec_style_layer[i])
        fig = plt.figure()
        plt.plot(var_ratio)
        plt.ylabel('var ratio')
        plt.xlabel('sum %g' % sum(var_ratio))
        fig.savefig('output/grid/var-ratio-%s.jpg' % rec_style_layer[i])
        plt.close(fig)


def imread(file_name):
    """ load image and cast type to float """
    return scipy.misc.imread(file_name).astype(np.float)

def imsave(file_name, image):
    """ clip image to range 0-255, cast type to uint8 and save """
    image = np.clip(image, 0, 255).astype(np.uint8)
    scipy.misc.imsave(file_name, image)

if __name__ == '__main__':
    main()
