import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
for vgg_model in range(10):
    import numpy as np
    import os
    import tensorflow.keras
    from tensorflow.keras.callbacks import CSVLogger
    # list of classes for each model
    class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
    # paths to Imagenet images
    train_dir  = "/home/daniel/replay/imagenet_split/train"
    validation_dir = "/home/daniel/replay/imagenet_split/val"
    test_dir = "/home/daniel/replay/imagenet_split/test"
    # directory where model will be saved
    checkpoint_directory_base = '/home/daniel/replay/01_baseline_training/checkpoint/'
    checkpoint_directory_full=os.path.join(checkpoint_directory_base,str(vgg_model))
    if not os.path.exists(checkpoint_directory_full):
        os.makedirs(checkpoint_directory_full)
    checkpoint_filename= os.path.join(checkpoint_directory_full, 'best_model.h5')
    history_directory_base = checkpoint_directory_full
    history_filename = os.path.join(history_directory_base, 'history_log.csv')
    # build lists of image paths and labels
    train_sets=[]
    for dp, dn, fn in os.walk(train_dir):
        dn[:] = [d for d in dn if d in class_list]
        for f in fn:    
            train_sets.append((os.path.join(dp, f), dp.split('/')[-1]))
    x_train, y_cls_train = zip(*train_sets)
    folder_numbers=list(range(10))
    y_classes = {class_list[i]:folder_numbers[i] for i in range(10)}
    y_train = [y_classes[y] for y in y_cls_train]
    y_train_oh = tensorflow.keras.utils.to_categorical(np.copy(y_train))
    val_sets=[]
    for dp, dn, fn in os.walk(validation_dir):
        dn[:] = [d for d in dn if d in class_list]
        for f in fn:    
            val_sets.append((os.path.join(dp, f), dp.split('/')[-1]))
    x_val, y_cls_val = zip(*val_sets)
    y_val = [y_classes[y] for y in y_cls_val]
    y_val_oh = tensorflow.keras.utils.to_categorical(np.copy(y_val))
    test_sets=[]
    for dp, dn, fn in os.walk(test_dir):
        dn[:] = [d for d in dn if d in class_list]
        for f in fn:
            test_sets.append((os.path.join(dp, f), dp.split('/')[-1]))
    x_test, y_cls_test = zip(*test_sets)
    y_test = [y_classes[y] for y in y_cls_test]
    y_test_oh = tensorflow.keras.utils.to_categorical(np.copy(y_test))
    type(np.asarray([0, 1, 2]))
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array  
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import math
    # set up data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='reflect')
    batch_size_train=36
    batch_size_val=26
    batch_size_test=25
    # build the sequence generators for training
    class ImagenetTrainSequence(tensorflow.keras.utils.Sequence):
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size
            self.shuffle_order = np.arange(len(self.x))
            np.random.shuffle(self.shuffle_order)
            self.x_shuf = self.x[self.shuffle_order]
            self.y_shuf = self.y[self.shuffle_order]
        def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)
        def __getitem__(self, idx):
            batch_x = self.x_shuf[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y_shuf[idx * self.batch_size:(idx + 1) * self.batch_size]
            return np.array([
                preprocess_input(datagen.apply_transform(img_to_array(load_img(file_name,target_size=(224, 224))),
                                                        (datagen.get_random_transform((224,224,3)))))
                    for file_name in batch_x]), np.array(batch_y)
        def on_epoch_end(self):
            np.random.shuffle(self.shuffle_order)
    # validation data is not augmented
    class ImagenetValSequence(tensorflow.keras.utils.Sequence):
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size
            self.shuffle_order = np.arange(len(self.x))
            np.random.shuffle(self.shuffle_order)
            self.x_shuf = self.x[self.shuffle_order]
            self.y_shuf = self.y[self.shuffle_order]
        def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)
        def __getitem__(self, idx):
            batch_x = self.x_shuf[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y_shuf[idx * self.batch_size:(idx + 1) * self.batch_size]
            return np.array([
                preprocess_input((img_to_array(load_img(file_name,target_size=(224, 224)))))
                    for file_name in batch_x]), np.array(batch_y)
        def on_epoch_end(self):
            np.random.shuffle(self.shuffle_order)        
    class ImagenetTestSequence(tensorflow.keras.utils.Sequence):
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size
            self.shuffle_order = np.arange(len(self.x))
            np.random.shuffle(self.shuffle_order)
            self.x_shuf = self.x[self.shuffle_order]
            self.y_shuf = self.y[self.shuffle_order]
        def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)
        def __getitem__(self, idx):
            batch_x = self.x_shuf[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y_shuf[idx * self.batch_size:(idx + 1) * self.batch_size]
            return np.array([
                preprocess_input((img_to_array(load_img(file_name,target_size=(224, 224)))))
                    for file_name in batch_x]), np.array(batch_y)
        def on_epoch_end(self):
            np.random.shuffle(self.shuffle_order)    
    train_seq = ImagenetTrainSequence(np.asarray(x_train),
                                 np.asarray(y_train_oh), 
                                 batch_size=batch_size_train)
    val_seq = ImagenetValSequence(np.asarray(x_val),
                               np.asarray(y_val_oh), 
                               batch_size=batch_size_val)
    test_seq = ImagenetTestSequence(np.asarray(x_test),
                               np.asarray(y_test_oh), 
                               batch_size=batch_size_test)
    val_seq.__getitem__(0)
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.layers import BatchNormalization    
    from tensorflow.keras.models import load_model
    # load the respective model
    original_model_save_filepath_base='/home/daniel/replay/02_replay_training/original_models/'+str(vgg_model)
    original_model_save_filepath_full=original_model_save_filepath_base+'/original_vgg.h5'
    model=load_model(original_model_save_filepath_full,custom_objects=None,compile=True)
    model.summary()
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras import optimizers
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint
    # save the best model based on minimum validation loss
    checkpoint = ModelCheckpoint(checkpoint_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # log the training and validation metrics    
    csv_logger = CSVLogger(history_filename, append=True, separator=';')
    from tensorflow.keras.callbacks import Callback
    # record the performance on the test set after each epoch
    class TestMetricCallback(Callback):
        def __init__(self, model):
            self.model = model
        def on_epoch_end(self, epoch,logs=None):
            test_metrics=model.evaluate(test_seq)
            print('test_metrics',test_metrics)
            def append_list_as_row(file_name, list_of_elem):
                with open(file_name, 'a+', newline='') as write_obj:
                    from csv import writer
                    csv_writer = writer(write_obj)
                    csv_writer.writerow(list_of_elem)
            test_metrics_filepath_base=os.path.join('/home/daniel/replay/01_baseline_training/checkpoint/',str(vgg_model))
            test_metrics_filepath=os.path.join(test_metrics_filepath_base,'test_validation_metrics.csv')
            append_list_as_row(test_metrics_filepath,test_metrics)
    test_metric_callback=TestMetricCallback(model)
    step_size_train = len(train_sets) // batch_size_train
    step_size_val = len(val_sets) // batch_size_val
    print ("train step size:", step_size_train)
    print ("train step val:", step_size_val)
    # epochs differs depending on experiment or two
    epochs=10
    # train the model
    history = model.fit(train_seq,
                            steps_per_epoch=step_size_train,
                            epochs=epochs,
                            validation_data=val_seq,
                            validation_steps=step_size_val,
                            callbacks=[csv_logger,checkpoint,test_metric_callback],
                            max_queue_size=16,
                            workers=8,
                            verbose=1,
                            )
    history.params.items()
    os.chdir(checkpoint_directory_full)
    import pickle
    with open('retrain_vgg.pickle','wb') as f:
         pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    model.save('current_model.h5')
    # clear gpu memory
    from tensorflow.compat.v1.keras.backend import set_session
    from tensorflow.compat.v1.keras.backend import clear_session
    from tensorflow.compat.v1.keras.backend import get_session
    import tensorflow
    import gc
    def reset_keras():
        sess = get_session()
        clear_session()
        sess.close()
        sess = get_session()
        try:
            del base_model
        except:
            pass
        try:
            del model 
        except:
            pass
        print(gc.collect()) 
        config = tensorflow.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        set_session(tensorflow.compat.v1.Session(config=config))
    reset_keras()
