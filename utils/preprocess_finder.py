import keras
import kerasapps

# def preprocess_finder(verbose=True):
def finder(context, verbose=True):
    if context.startswith('xception'):
        if verbose:
            print('----> USING {} native preprocessing'.format(context))
        return keras.applications.xception.preprocess_input
    elif context.startswith('resnet50'):
        if verbose:
            print('----> USING {} native preprocessing'.format(context))
        return keras.applications.resnet50.preprocess_input
    elif context.startswith('inception_resnet_v2'):
        if verbose:
            print('----> USING {} native preprocessing'.format(context))
        return keras.applications.inception_resnet_v2.preprocess_input
    elif context.startswith('inception_v3'):
        if verbose:
            print('----> USING {} native preprocessing'.format(context))
        return keras.applications.inception_v3.preprocess_input
    elif context.startswith('mobilenet_v2'):
        if verbose:
            print('----> USING {} native preprocessing'.format(context))
        return keras.applications.mobilenet_v2.preprocess_input
    elif context.startswith( ('resnet152_v2', 'resnet101_v2') ):
        if verbose:
            print('----> USING {} native preprocessing'.format(context))
        return kerasapps.keras_applications.resnet_v2.preprocess_input
    elif context.startswith('resnet'):
        if verbose:
            print('----> USING {} native preprocessing'.format(context))
        return kerasapps.keras_applications.resnet.preprocess_input
    else:
        if verbose:
            print('----> USING 1 / 255. as preprocessing')
        return lambda x : x / 255.
    # return finder