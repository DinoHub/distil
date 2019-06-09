import os
import keras
import pickle
import shutil
import random

import pandas as pd
import numpy as np
import keras.backend as K

from collections import defaultdict

from utils.preprocess_finder import finder

import kerasapps.keras_applications
kerasapps.keras_applications.set_keras_submodules(backend=keras.backend, layers=keras.layers,models=keras.models, utils=keras.utils)

from keras import optimizers

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.models import Model, load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler

# bs 32
# progressive scaling helps
def get_xception(num_classes, verbose=True):
    from keras.applications.xception import Xception
    # base_model = Xception(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = Xception(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (299, 299), model

# Native preprocessing
# No progressive scaling
def get_resnet50(num_classes, verbose=True):
    from keras.applications.resnet50 import ResNet50
    # base_model = ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

def get_mobilenet_v2(num_classes, verbose=True):
    from kerasapps.keras_applications.mobilenet_v2 import MobileNetV2
    # from keras.applications.mobilenet_v2 import MobileNetV2
    # base_model = MobileNetV2(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = MobileNetV2(weights='imagenet', alpha=1.4, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

def get_model(context, num_classes, verbose=True):
    if context.startswith('xception'):
        return get_xception( num_classes, verbose )
    elif context.startswith('resnet50'):
        return get_resnet50(num_classes, verbose)
    elif context.startswith('mobilenet_v2'):
        return get_mobilenet_v2(num_classes, verbose)


def folder_to_df(source_folder, target_frame, filenames, soft_preds):
    sorted_classes = sorted(list(set([f.split('_')[0].lower() for f in os.listdir(source_folder)])))
    num_classes = len(sorted_classes)

    label_dict = {}
    for i,c in enumerate(sorted_classes):
        label_dict[c] = i

    fn_softlabel_dict = dict( zip( filenames, soft_preds ) )
    rows = []

    for im_fp in os.listdir( source_folder ):
        payload = [im_fp]
        softlabels = fn_softlabel_dict[im_fp]
        for sl in softlabels:
            payload.append(sl)
        payload.append( label_dict[im_fp.split('_')[0].lower()] )
        rows.append( payload )

    cols = ['im_path']
    cols += ['soft{}'.format(k) for k in range(num_classes)]
    cols += ['labelidx']

    df = pd.DataFrame(rows, columns=cols)

    df = pd.concat( [df,pd.get_dummies( df['labelidx'] )], axis=1 )

    df = df.drop( ['labelidx'], axis=1 )

    df.to_csv( target_frame, index=False )

    return df

def groupby_pids(pose_dirp):
    pose_dict = defaultdict(list)
    img_names = os.listdir( pose_dirp )
    for img_name in img_names:
        _, pid, _ = img_name.split('_')
        pose_dict[pid].append( img_name )
    return pose_dict

# splits source into target/train and target/val, by pids
def generate_train_val_split(source_folder, target_folder, ratio=0.1):
    if os.path.exists( target_folder ):
        shutil.rmtree( target_folder )

    for pose in os.listdir( source_folder ):
        pose_dirp = os.path.join( source_folder, pose )
        posewise_imgs = len(list(os.listdir(pose_dirp)))
        pose_dict = groupby_pids( pose_dirp )
        # shuff_list = list(os.listdir( pose_dirp ))
        shuff_list = list(pose_dict.keys())
        random.shuffle( shuff_list )

        idx = 0

        target_pose_dir = '{}/val'.format(target_folder)
        if not os.path.exists( target_pose_dir ):
            os.makedirs( target_pose_dir )

        target_count = int( posewise_imgs * ratio )

        curr_count = 0
        while curr_count < target_count and idx < len(shuff_list):
            pid = shuff_list[idx]

            src_tgt_pairs = [(os.path.join(pose_dirp, f), os.path.join(target_pose_dir, f)) for f in pose_dict[pid]]
            curr_count += len( src_tgt_pairs )
            # perform a write to target_pose_dir
            for src_fp, tgt_fp in src_tgt_pairs:
                # print('moving {} to {}'.format(src_fp, tgt_fp))
                shutil.copy( src_fp, tgt_fp )
            idx += 1

        target_pose_dir = '{}/train'.format(target_folder)
        if not os.path.exists( target_pose_dir ):
            os.makedirs( target_pose_dir )
        while idx < len(shuff_list):
            pid = shuff_list[idx]

            src_tgt_pairs = [(os.path.join(pose_dirp, f), os.path.join(target_pose_dir, f)) for f in pose_dict[pid]]
            curr_count += len( src_tgt_pairs )
            # perform a write to target_pose_dir
            for src_fp, tgt_fp in src_tgt_pairs:
                # print('moving {} to {}'.format(src_fp, tgt_fp))
                shutil.copy( src_fp, tgt_fp )
            idx += 1

def train_at_scale(model, scale, csvLogger, valLossCP, valAccCP, tbCallback, lrCallback, kwargs, bs, train_folder, val_folder, n_epochs):
    # more intense augmentations
    train_datagen = ImageDataGenerator(
            rotation_range=45,#in deg
            brightness_range= [0.5,1.5],
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            **kwargs)

    val_datagen = ImageDataGenerator(**kwargs)

    train_generator = train_datagen.flow_from_directory(
            train_folder,
            target_size=scale,
            batch_size=bs,
            class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            val_folder,
            target_size=scale,
            batch_size=bs,
            class_mode='categorical')

    if lrCallback is not None:
        all_callbacks = [csvLogger, valLossCP, valAccCP, tbCallback, lrCallback]
    else:
        all_callbacks = [csvLogger, valLossCP, valAccCP, tbCallback]

    model.fit_generator(train_generator,
            steps_per_epoch=train_generator.samples // bs,
            epochs=n_epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // bs,
            callbacks=all_callbacks)

def scheduler(epoch_idx):
    if epoch_idx < 3:
        return 0.01
    elif epoch_idx < 10:
        return 0.001
    return max( 8e-5, 0.0001 - epoch_idx * 1e-6 )

def train_from_scratch(source_folder, target_folder, contexts, num_classes, save_at_end=False, ngpus=1):
    # finder = preprocess_finder()
    train_folder = os.path.join(target_folder, 'train')
    val_folder = os.path.join(target_folder, 'val')
    for context in contexts:
        # Each round, we train on a different split
        generate_train_val_split(source_folder, target_folder, ratio=0.1)

        if not os.path.exists( 'models/{}'.format(context) ):
            os.makedirs( 'models/{}'.format(context) )

        bs, target_size, model = get_model(context, num_classes)
        if ngpus > 1:
            model = multi_gpu_model(model, gpus=ngpus, cpu_relocation=True)
        # model.compile(optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        csvLogger = CSVLogger('logs/{}.log'.format(context))
        valLossCP = ModelCheckpoint('models/{}/{}_loss.hdf5'.format(context, context), save_best_only=True)
        valAccCP = ModelCheckpoint('models/{}/{}_acc.hdf5'.format(context, context), monitor='val_acc', save_best_only=True)
        tbCallback = TensorBoard( log_dir='./{}_tblogs'.format(context), histogram_freq=0, write_graph=True, write_images=True )
        # lrCallback = LearningRateScheduler(scheduler, verbose=1)
        lrCallback = None

        # progressive scaling
        # scales = [(75,75), (150,150), (224,224)]
        # epochses = [10, 10, 200]
        scales = [(224,224)]
        epochses = [120]
        for scale, epochs in zip(scales, epochses):
            train_at_scale(model, scale, csvLogger, valLossCP, valAccCP, tbCallback, lrCallback, {'preprocessing_function': finder(context)}, bs, train_folder, val_folder, epochs)

        if save_at_end:
            model.save('models/{}/{}_last.hdf5'.format(context,context))

        del model

if __name__ == '__main__':
    # data_folder = 'data'
    # df = folder_to_df( data_folder )
    # df.to_csv( 'data.csv', index=False )
    # exit()
    fn_list_pickle = 'full_cropped_filenames.p'
    softlabels = 'full_cropped_softlabels.txt'
    source_folder = 'data/full_cropped'
    split_dir = 'data/full_cropped_split'
    train_df_fp = 'frames/train.csv'
    train_data = os.path.join(split_dir, 'train')
    val_df_fp = 'frames/val.csv'
    val_data = os.path.join(split_dir, 'val')

    with open(fn_list_pickle, 'rb') as f:
        filenames = pickle.load(f)
    soft_preds = np.loadtxt( softlabels )

    # source_folder = 'data/full_cropped'
    # target_folder= 'data/full_cropped_split'
    # generate_train_val_split(source_folder, target_folder)

    # folder_to_df( os.path.join(split_dir, 'val'), val_df, filenames, soft_preds )
    # exit()

    n_epochs = 250
    # target_size = (224,224)
    batch_size = 32
    num_classes = 16
    alpha = 0.5
    x_col = 'im_path'
    y_col = ['soft{}'.format(k) for k in range(num_classes)] + ['{}'.format(k) for k in range(num_classes)]

    context = 'mobilenet_v2_distilled'
    if not os.path.exists( 'models/{}'.format(context) ):
        os.makedirs( 'models/{}'.format(context) )

    # df = pd.read_csv( './data2.csv' )
    bs, target_size, model = get_model(context, num_classes)

    generate_train_val_split(source_folder, split_dir)
    folder_to_df( train_data, train_df_fp, filenames, soft_preds )
    folder_to_df( val_data, val_df_fp, filenames, soft_preds )

    train_df = pd.read_csv( train_df_fp )
    train_datagen = ImageDataGenerator(
            rotation_range=45,#in deg
            brightness_range= [0.5,1.5],
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        preprocessing_function=finder(context))
    train_generator = train_datagen.flow_from_dataframe( 
        train_df, 
        directory=train_data,
        x_col=x_col,
        y_col=y_col,
        class_mode='other',
        batch_size=batch_size,
        target_size=target_size)

    val_df = pd.read_csv( val_df_fp )
    val_datagen = ImageDataGenerator(preprocessing_function=finder(context))
    val_generator = val_datagen.flow_from_dataframe( 
        val_df, 
        directory=val_data,
        x_col=x_col,
        y_col=y_col,
        class_mode='other',
        batch_size=batch_size,
        target_size=target_size)

    def custom_loss(ytrue, ypred):
        return alpha*K.categorical_crossentropy(ytrue[:,:num_classes], ypred) + (1. - alpha)*K.categorical_crossentropy(ytrue[:,num_classes:], ypred)

    model.compile(optimizer='adam',
                  loss=custom_loss)
    csvLogger = CSVLogger('logs/{}.log'.format(context))
    valLossCP = ModelCheckpoint('models/{}/{}_loss.hdf5'.format(context, context), save_best_only=True)

    model.fit_generator(train_generator,
            steps_per_epoch=train_generator.samples // bs + 1,
            epochs=n_epochs,
            validation_data=val_generator,
            validation_steps=val_generator.samples // bs + 1,
            callbacks=[csvLogger, valLossCP])
