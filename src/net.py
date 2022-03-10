from tf2cv.model_provider import get_model as tf2cv_get_model

resnet = tf2cv_get_model('resnet56_cifar10', pretrained=True, data_format='channels_last')
preresnet = tf2cv_get_model('preresnet56_cifar10', pretrained=True, data_format='channels_last')
resnext = tf2cv_get_model('resnext29_16x64d_cifar10', pretrained=True, data_format='channels_last')
seresnet = tf2cv_get_model('seresnet56_cifar10', pretrained=True, data_format='channels_last')
densenet = tf2cv_get_model('densenet40_k12_cifar10', pretrained=True, data_format='channels_last')
