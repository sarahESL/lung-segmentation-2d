import argparse
from image_gen import ImageDataGenerator
from load_data import loadDataMontgomery, loadDataJSRT
from build_model import build_UNet2D_4L
import keras
from tensorflow import convert_to_tensor
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow import InteractiveSession
# import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # index of the gp


def train(dataset, save_grads):
    print("GPU Available: ", tf.test.is_gpu_available())

    if(dataset == "JSRT"):
        csv_path = '/home/sedigheh/lung_segmentation/dataset/' + \
            'JSRT/preprocessed_org_with_mask/idx.csv'
    elif(dataset == "Montgomery"):
        csv_path = '/home/sedigheh/lung_segmentation/dataset/' + \
            'MontgomerySet/mixed_images_masks/idx.csv'
    else:
        """ In case needed for CT images"""
        # csv_path = '/home/sedigheh/lung_segmentation/dataset/' + \
        # 'CT-kaggle-lung/mixed/idx.csv'
        raise ValueError("Dataset not supported!")

    path = csv_path[:csv_path.rfind('/')] + '/'

    original_df = pd.read_csv(csv_path, header=None)
    # Shuffle rows in dataframe. Random state is set for reproducibility.
    df = original_df.sample(frac=1, random_state=23)

    n_train = int(0.8 * len(df))
    df_train = df[:n_train]
    df_val = df[n_train:]

    # Load training and validation data
    im_shape = (256, 256)

    print("Loading data...")
    if(dataset == "JSRT"):
        X_train, y_train = loadDataJSRT(df_train, path, im_shape)
        X_val, y_val = loadDataJSRT(df_val, path, im_shape)
    elif(dataset == "Montgomery"):
        X_train, y_train = loadDataMontgomery(df_train, path, im_shape)
        X_val, y_val = loadDataMontgomery(df_val, path, im_shape)
    else:
        raise ValueError("Dataset not supported!")

    # Build model
    org_shape = X_train[0].shape
    inp_shape = (org_shape[0], org_shape[1], org_shape[2])
    print("Building model...")
    UNet = build_UNet2D_4L(inp_shape)
    UNet.compile(optimizer='adam',
                 loss='binary_crossentropy', metrics=['accuracy'])

    # Visualize model
    # plot_model(UNet, 'model.png', show_shapes=True)

    ########################################################################
    print("**************Fitting model****************")
    model_file_format = 'model.{epoch:03d}.hdf5'
    checkpointer = ModelCheckpoint(model_file_format, period=10)

    print("ImageGenerator....")
    train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)

    test_gen = ImageDataGenerator(rescale=1.)

    batch_size = 8
    UNet.fit_generator(train_gen.flow(X_val, y_val, batch_size),
                       steps_per_epoch=(X_train.shape[0] +
                                        batch_size - 1) // batch_size,
                       epochs=100,
                       callbacks=[checkpointer],
                       validation_data=test_gen.flow(X_val, y_val),
                       validation_steps=(X_val.shape[0] +
                                         batch_size - 1) // batch_size)
    del X_train, X_val, y_train, y_val

    print("**************Computing gradients****************")

    sess = InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    imagename_gradsign_dict = {}
    for index, _ in original_df.iterrows():
        row = original_df.iloc[[index]]
        imagename = str(row.iloc[0, 0])
        X, y = loadDataJSRT(row, path, im_shape)
        y_tensor = convert_to_tensor(y, dtype=tf.float32)
        loss = keras.losses.binary_crossentropy(y_tensor, UNet.output)
        gradients = keras.backend.gradients(loss, UNet.input)
        gradients = sess.run(gradients[0], feed_dict={UNet.input: X})
        gradients_sign = sess.run(keras.backend.sign(gradients))
        imagename_gradsign_dict[imagename] = gradients_sign
    print("Save image_gradsign_dict to file...")
    if(dataset == "JSRT"):
        np.save("/home/sedigheh/lung_segmentation/gradient_signs/" +
                "jsrt_imagename_gradsign_dict.npy", imagename_gradsign_dict)
    elif(dataset == "Montgomery"):
        np.save("/home/sedigheh/lung_segmentation/gradient_signs/" +
                "montgomery_imagename_gradsign_dict.npy",
                imagename_gradsign_dict)
    else:
        raise ValueError("Dataset not supported!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset to use for training." +
                        "Supported: JSRT and Montgomery")
    parser.add_argument("--save_gradients", default=True,
                        help="If compute the gadient w.r.t each input image " +
                        "and save to npy file. Deaults to True. False is " +
                        "not supported.")
    args = parser.parse_args()
    dataset = args.dataset
    save_grads = args.save_gradients
    train(dataset, save_grads)
