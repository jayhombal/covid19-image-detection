
# common imports
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# To Avoid GPU errors
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"


#callback setup
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau






def get_base_model(model_name :str = 'ResNet50V2', freeze_layers:bool = False, image_shape : tuple = (224,224,3)):
    """
    Returns the base model with all frozen layers after removing the top layer, 
    Args:
        model_name (str): model_name values pretrained_models = ['ResNet50V2', 'MobileNetV2', 'VGG16']
    """
    if model_name == 'ResNet50V2' :
        print(f"Downloading ResNet50V2")
        base_model = tf.keras.applications.ResNet50V2(input_shape=image_shape,
                                               include_top=False,
                                               weights='imagenet')
    elif model_name == 'MobileNetV2' :
        print(f"Downloading MobileNetV2")
        base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                               include_top=False,
                                               weights='imagenet')
    elif model_name == 'VGG16' :
        print(f"Downloading VGG16")
        base_model = tf.keras.applications.VGG16(input_shape=image_shape,
                                               include_top=False,
                                               weights='imagenet')
    if freeze_layers == True:
        for layer in base_model.layers:
            layer.trainable = False
            #assert layer.trainable is False
    
    return base_model

 
def compile_classifier(model, learning_rate, optimizer = 'Adam' , activation_type:str = 'softmax'):
    """[summary]
    Returns a compiled model, this method can be extended to use other optimizers
    Args:
        model ([tensorflow.keras.Model]): classifier model
        learning_rate ([float]): [description]
        optimizer ([tensorflow.keras.optimizers]): optimizer
    """
    if optimizer == 'Adam':
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True,
            name="Adam"
            )
    if activation_type == 'softmax':
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = ['accuracy']
    elif activation_type == 'sigmoid':
        loss = tf.keras.losses.binary_crossentropy,
        metrics = [keras.metrics.MAE, 
            keras.metrics.AUC(name='auc',multi_label=True),
            keras.metrics.BinaryAccuracy(threshold=0.65),
            keras.metrics.FalseNegatives(),
            keras.metrics.FalsePositives()]
    #compile model

    model.compile(optimizer=optimizer, loss = loss, metrics = metrics)
    
    return model

def get_base_model_with_new_toplayer(base_model, 
                                     freeze_layers: bool = False, 
                                     num_classes: int = 15,
                                     activation_func: str = 'softmax',  # softmax or sigmoid
                                     learning_rate: float = 0.01,
                                     input_shape : tuple = (224,224,3)):
    """ 
    Returns a model with a baseline architecture of ResNet50V2, VGG16, MobileNetV2 models
    Args:
        base_model ([keras.Model]): base_model
        num_classes ([int]) : number classes
    """
    print(f"learning rate {learning_rate}")
    base_model = get_base_model(base_model,freeze_layers,input_shape)
    head_model = base_model.output
    head_model = keras.layers.Flatten(name="flatten")(head_model)
    # head_model = keras.layers.Dense(256, activation="relu")(head_model)
    # head_model = keras.layers.Dropout(0.3)(head_model)
    # head_model = keras.layers.Dense(128, activation="relu")(head_model)
    # head_model = keras.layers.Dropout(0.3)(head_model)
    # head_model = keras.layers.Dense(64, activation="relu")(head_model)
    # head_model = keras.layers.Dense(64, activation='relu')(head_model)   #Added Starts Here
    # head_model = keras.layers.BatchNormalization()(head_model)
    # head_model = keras.layers.Dropout(rate=0.5)(head_model)
    # head_model = keras.layers.Dense(64, activation='relu')(head_model)
    # head_model = keras.layers.BatchNormalization()(head_model)
    # head_model = keras.layers.Dropout(rate=0.5)(head_model)
    # head_model = keras.layers.Dense(32, activation='relu')(head_model)
    # head_model = keras.layers.BatchNormalization()(head_model)
    # head_model = keras.layers.Dropout(rate=0.5)(head_model)
    # head_model = keras.layers.Dense(32, activation='relu')(head_model)
    # head_model = keras.layers.BatchNormalization()(head_model)
    # head_model = keras.layers.Dropout(rate=0.5)(head_model)                    #Added Ends Here
    head_model = keras.layers.Dense(num_classes,activation=activation_func)(head_model)
    model = keras.Model(inputs=base_model.input, outputs=head_model)
    model = compile_classifier(model, learning_rate, optimizer='Adam', activation_type=activation_func)
    return model


def fine_tune_model(model, learning_rate =0.00001, optimizer = 'Adam',  fine_tune_at_layer:int=178, activation_func: str = 'softmax'):
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[fine_tune_at_layer:]:
        layer.trainable =  True
    compile_classifier(model, learning_rate = learning_rate, optimizer=optimizer, activation_type = activation_func)
    return model




def fit_model(model,
              train_ds,
              validation_ds, 
              num_epochs: int = 20, 
              batch_size: int = 32, 
              checkpoint_filepath:str = 'models/my_model.h5', 
              logs_dir: str = 'logs/fit'):
    
    
    checkpoint = ModelCheckpoint(checkpoint_filepath, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min')

    early = EarlyStopping(monitor="val_loss", min_delta = 1e-4, patience = 3, mode = 'min', 
                    restore_best_weights = True, verbose = 1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 2 ) #, verbose = 1, 
                                #min_delta = 1e-4, min_lr = 1e-6, mode = 'min', cooldown=1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)


    callbacks = [checkpoint, early, reduce_lr, tensorboard_callback]

    history = model.fit(train_ds,
                    epochs=num_epochs,
                    validation_data=validation_ds,
                    steps_per_epoch = len(train_ds)//batch_size,#steps_per_epoch = 100, 
                    validation_steps=len(validation_ds)//batch_size, #validation_steps= 25, 
                    callbacks=callbacks)
    return history

def plot_epocs_vs_val_loss(history):
    plt.figure(figsize = (12, 6))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot( history.history["loss"], label = "Training Loss", marker='o')
    plt.plot( history.history["val_loss"], label = "Validation Loss", marker='+')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_epocs_vs_auc(history):
    plt.figure(figsize = (12, 6))
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.plot( history.history["auc"], label = "Training AUC" , marker='o')
    plt.plot( history.history["val_auc"], label = "Validation AUC", marker='+')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_accuracy_and_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training & Validation - Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.ylim([0,max(plt.ylim())])
    plt.title('Training & Validation - Loss')
    plt.xlabel('Epoch')
    plt.show()


from keras.applications.resnet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


def get_image_data_generator(train_df, lables, batch_size: int = 32, shuffle: bool = True, seed: int = 42, image_size : tuple = (224,224)):
    image_data_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=True, #Boolean. Set each sample mean to 0.
    samplewise_std_normalization = False, #Boolean. Divide each input by its std.
    featurewise_std_normalization=False, # divide inputs by std of the dataset
    horizontal_flip = True, #Boolean. Randomly flip inputs horizontally.
    vertical_flip = False,  #Boolean. Randomly flip inputs vertically.
    zca_whitening=False,  # apply ZCA whitening
    height_shift_range= 0.05, #float: fraction of total height, if < 1, or pixels if >= 1.
    width_shift_range=0.1,  #float: fraction of total height, if < 1, or pixels if >= 1.
    rotation_range=20, #Int. Degree range for random rotations. 0 -180 degrees
    shear_range = 0.1, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
    fill_mode = 'nearest', #One of {"constant", "nearest", "reflect" or "wrap"}. Default is 'nearest'. 
    zoom_range=0.15) #Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]

    #Takes the dataframe and the path to a directory + generates batches.
    img_generator = image_data_gen.flow_from_dataframe(
                dataframe=train_df,
                directory=None, #string, path to the directory to read images from. 
                                #If None, data in x_col column should be absolute paths.
                x_col='path', #string, column in dataframe that contains the filenames (or absolute paths if directory is None).
                y_col='finding_label', #string or list, column/s in dataframe that has the target data.
        
                class_mode="categorical", #one of "binary", "categorical", "input", "multi_output", "raw", sparse" or None. Default: "categorical". 
                                # Mode for yielding the targets: "raw": numpy array of values in y_col column(s),
                classes=lables,
                #color_mode='grayscale',
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                target_size= image_size)
    return img_generator

