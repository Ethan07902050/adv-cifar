import sys
import numpy as np
import os
import pickle
import tensorflow as tf
import net
from imageio import imread, imwrite

img_dir = sys.argv[1]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
max_epsilon = 8
eps = 2 * max_epsilon / 255.0
num_iter = 10
momentum = 0.5
output_dir = f'./adv_imgs'

def load_images():
    
    batch_shape = (10, 32, 32, 3)
    
    for sub_dir in os.listdir(img_dir):
        
        path = os.path.join(img_dir, sub_dir)
        label = class_names.index(sub_dir)
        images = np.zeros(batch_shape)

        for img_path in os.listdir(path):

            idx = int(img_path[-5])
            img_path = os.path.join(path, img_path)
            img = imread(img_path)
            images[idx-1, :, :, :] = img
                                    
        yield images, label


def save_images(images, class_name, x_min, x_max):
    dir_name = os.path.join(output_dir, class_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    images = tf.multiply(tf.add(images, 1.0), 127.5)
    images = tf.clip_by_value(images, x_min, x_max)
    images = tf.cast(images, dtype=tf.uint8)
    
    for i in range(10):
        img_path = os.path.join(dir_name, class_name + str(i+1) + '.png')
        imwrite(img_path, images[i, :, :, :], format='png')
        

def MI_FGSM(x, y, it, grads):
    
    alpha = eps / num_iter
    num_classes = 10

    one_hot = tf.one_hot(label, num_classes)
    one_hot = tf.expand_dims(one_hot, axis=0)
    one_hot = tf.repeat(one_hot, repeats=[10], axis=0)

    with tf.GradientTape() as g:
        g.watch(x)
        resnet_logits = net.resnet(x)
        preresnet_logits = net.preresnet(x)
        resnext_logits = net.resnext(x)
        seresnet_logits = net.seresnet(x)
        densenet_logits = net.densenet(x)
        logits = (resnet_logits + preresnet_logits + resnext_logits + seresnet_logits + densenet_logits) * 0.2
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(one_hot, logits)

    noise = g.gradient(cross_entropy, x)
    noise /= tf.reduce_mean(tf.abs(noise), [1,2,3], keepdims=True)
    noise = momentum * grads + noise

    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, -1.0, 1.0)
    it = tf.add(it, 1)
    return x, y, it, noise
    

def cond(x, y, it, grads):
    return tf.less(it, num_iter)


if __name__ == '__main__':

    for i, (images, label) in enumerate(load_images()):
        x_max = tf.clip_by_value(images + max_epsilon, 0, 255)
        x_min = tf.clip_by_value(images - max_epsilon, 0, 255)
        images = tf.subtract(tf.divide(images, 127.5), 1.0)

        x_max = tf.cast(x_max, tf.float32)
        x_min = tf.cast(x_min, tf.float32)
        images = tf.cast(images, tf.float32)
        
        grads = np.zeros(images.shape)
        it = tf.constant(0)
        x_adv, _, _, _, = tf.while_loop(cond, MI_FGSM, [images, label, it, grads])
        save_images(x_adv, class_names[label], x_min, x_max)
