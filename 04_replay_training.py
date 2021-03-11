import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array  
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
for vgg_model in range(0,5): 
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.layers import BatchNormalization    
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras import optimizers
    # path to the original models to be trained
    original_model_save_filepath_base='/home/daniel/replay/replay_training/original_models/'+str(vgg_model)
    original_model_save_filepath_full=original_model_save_filepath_base+'/original_vgg.h5'
    current_model_save_filepath_base='/home/daniel/replay/replay_training/checkpoint/'+str(vgg_model)
    if not os.path.exists(current_model_save_filepath_base):
        os.makedirs(current_model_save_filepath_base)
    current_model_save_filepath_full=current_model_save_filepath_base+'/current_model.h5'
    # number of epochs changes depending on experiment one or two
    for training_epoch in range(30):
        # list of classes for each model
        class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
        prepare data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='reflect')
        # create folder for the augmented images
        for image_class in class_list:
            train_dir_augmented_base  = "/home/daniel/replay/replay_training/train_augmented/"
            train_dir_augmented_full = train_dir_augmented_base+image_class
            if not os.path.exists(train_dir_augmented_full):
                os.makedirs(train_dir_augmented_full)
            train_dir_base  = "/home/daniel/replay/imagenet_split/train/"
            train_dir_full = train_dir_base+image_class
            list_of_files=os.listdir(train_dir_full)
            list_of_files.sort()
            os.chdir(train_dir_full)
            # preprocess and augment the images
            for filename in list_of_files:
                temp_img=datagen.apply_transform(img_to_array(load_img(filename,target_size=(224, 224))),
                                                            (datagen.get_random_transform((224,224,3))))
                temp_img=temp_img.astype(np.uint8)
                temp_img = Image.fromarray(temp_img)
                save_filename=os.path.join(train_dir_augmented_full,filename)
                temp_img.save(save_filename)
        # to train the model, will be alternated with replay
        def training(training_epoch,original_model_save_filepath_full,current_model_save_filepath_full):
            import numpy as np
            import os
            import tensorflow.keras
            from tensorflow.keras.callbacks import CSVLogger
            from tensorflow.keras.models import load_model
            import math
            # start with the original model if it is the first epoch
            if training_epoch==0:
                model=load_model(original_model_save_filepath_full,custom_objects=None,compile=True)
            else:
                model=load_model(current_model_save_filepath_full,custom_objects=None,compile=True)
            class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
            # set the paths to the imagnet images
            train_dir  = "/home/daniel/replay/replay_training/train_augmented"
            validation_dir = "/home/daniel/replay/imagenet_split/val"
            test_dir = "/home/daniel/replay/imagenet_split/test"
            checkpoint_directory_base = '/home/daniel/replay/replay_training/checkpoint'
            checkpoint_directory_full=os.path.join(checkpoint_directory_base,str(vgg_model))
            if not os.path.exists(checkpoint_directory_full):
                os.makedirs(checkpoint_directory_full)
            # create the current checkpoing model path
            checkpoint_filename= os.path.join(checkpoint_directory_full, 'current_model.h5')
            history_directory_base = checkpoint_directory_full
            # create history file for logging training metrics
            history_filename = os.path.join(history_directory_base, 'history_log.csv')
            train_sets=[]
            # set up paths for 10 selected training classes
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
            # do the same for the validation and test images
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
            batch_size_train=36
            batch_size_val=26
            batch_size_test=25
            # set up sequence generator to supply model with training images
            # images are already augmented so do not need to be in this step
            class ImagenetSequence(tensorflow.keras.utils.Sequence):
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
            train_seq = ImagenetSequence(np.asarray(x_train),
                                         np.asarray(y_train_oh), 
                                         batch_size=batch_size_train)
            val_seq = ImagenetSequence(np.asarray(x_val),
                                       np.asarray(y_val_oh), 
                                       batch_size=batch_size_val)
            test_seq = ImagenetSequence(np.asarray(x_test),
                           np.asarray(y_test_oh), 
                           batch_size=batch_size_test)    
            csv_logger = CSVLogger(history_filename, append=True, separator=';')
            from tensorflow.keras.callbacks import Callback
            step_size_train = len(train_sets) // batch_size_train
            step_size_val = len(val_sets) // batch_size_val
            print ("train step size:", step_size_train)
            print ("train step val:", step_size_val)
            epochs=1
            history = model.fit(train_seq,
                                    steps_per_epoch=step_size_train,
                                    epochs=epochs,
                                    validation_data=val_seq,
                                    validation_steps=step_size_val,
                                    callbacks=[csv_logger],
                                    max_queue_size=16,
                                    workers=8,
                                    verbose=1,
                                    )
            history.params.items()
            model.save(current_model_save_filepath_full)
            # clear the gpu memory to deal with memory leak issue
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
                    del model 
                except:
                    pass
                print(gc.collect()) 
                config = tensorflow.compat.v1.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 1
                config.gpu_options.visible_device_list = "0"
                set_session(tensorflow.compat.v1.Session(config=config))
            reset_keras()
            # get the metrics on the training set and save to csv
            model=load_model(current_model_save_filepath_full,custom_objects=None,compile=True)
            test_metrics=model.evaluate(test_seq)
            print('test_metrics',test_metrics)
            def append_list_as_row(file_name, list_of_elem):
                with open(file_name, 'a+', newline='') as write_obj:
                    from csv import writer
                    csv_writer = writer(write_obj)
                    csv_writer.writerow(list_of_elem)
            test_metrics_filepath_base=os.path.join('/home/daniel/replay/replay_training/checkpoint/',str(vgg_model))
            test_metrics_filepath=os.path.join(test_metrics_filepath_base,'test_validation_metrics.csv')
            append_list_as_row(test_metrics_filepath,test_metrics)
            # save the best model if validation loss is lower than previous epoch
            # also, keep a record of the lowest validation loss so far
            current_loss=history.history['val_loss'][:]
            if training_epoch==0:
                current_loss_filepath=os.path.join(checkpoint_directory_full,'lowest_loss.npy')
                np.save(current_loss_filepath,current_loss)
                best_model_filepath=os.path.join(checkpoint_directory_full,'best_model.h5')
                model.save(best_model_filepath)
            else:
                current_loss_filepath=os.path.join(checkpoint_directory_full,'lowest_loss.npy')
                previous_lowest_loss=np.load(current_loss_filepath)
                if current_loss<previous_lowest_loss:
                    np.save(current_loss_filepath,current_loss)
                    best_model_filepath=os.path.join(checkpoint_directory_full,'best_model.h5')
                    model.save(best_model_filepath)
            # clear the gpu memory again
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
                    del model 
                except:
                    pass
                print(gc.collect())
                config = tensorflow.compat.v1.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 1
                config.gpu_options.visible_device_list = "0"
                set_session(tensorflow.compat.v1.Session(config=config))
            reset_keras()
        training(training_epoch,original_model_save_filepath_full,current_model_save_filepath_full)
        
        # get activations from the relevant layer by feeding images through
        # these will be used to generate the multivariate distribution to sample from
        # and create new samples. These will also be used as training images for
        # comparing generative with veridical replay
        def activations(current_model_save_filepath_full):
            import numpy as np
            from tensorflow.keras.applications.vgg16 import preprocess_input
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.models import load_model
            import keract
            class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
            # load the current model in training
            model=load_model(current_model_save_filepath_full,custom_objects=None, compile=True)
            for image_class in class_list:
                # use the augmented images
                train_dir_base="/home/daniel/replay/replay_training/train_augmented"
                train_dir_full=os.path.join(train_dir_base,image_class)
                os.chdir(train_dir_full)
                list_of_files=os.listdir(train_dir_full)
                list_of_files.sort()
                for files in list_of_files:
                    os.chdir(train_dir_full)
                    img = image.load_img(files, target_size=(224, 224))
                    img_tensor = image.img_to_array(img)
                    img_tensor = np.expand_dims(img_tensor, axis=0)
                    x_train=preprocess_input(img_tensor)
                    # the keract toolbox is used to get the relevant activations
                    # change "block4_pool" to any of the other pooling layers in the network
                    test_activation=keract.get_activations(model, x_train, layer_names='block4_pool', nodes_to_evaluate=None, output_format='simple', auto_compile=True)
                    test_activation_values=test_activation.values()
                    data = list(test_activation_values)
                    data_array = np.array(data)
                    data_array_squeezed=np.squeeze(data_array)
                    # saved as float16 to save disk space
                    data_array_squeezed=data_array_squeezed.astype('float16')
                    # save these activations
                    basedir='/home/daniel/replay/replay_training/train_augmented_activations'
                    activation_dir = os.path.join(basedir,image_class)
                    if not os.path.exists(activation_dir):
                        os.makedirs(activation_dir)
                    os.chdir(activation_dir)
                    extension='.npy'
                    data_title=files+extension
                    np.save(data_title, data_array_squeezed)
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
                    del model 
                except:
                    pass
                print(gc.collect()) 
                config = tensorflow.compat.v1.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 1
                config.gpu_options.visible_device_list = "0"
                set_session(tensorflow.compat.v1.Session(config=config))
            reset_keras()        
        activations(current_model_save_filepath_full)
        
        # get activations from the same layer in the network to be used as the validation set
        # during replay training. We will replay using the "imagined" samples but validate on real ones
        def val_activations(current_model_save_filepath_full):
            import numpy as np
            from tensorflow.keras.applications.vgg16 import preprocess_input
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.models import load_model
            import keract
            class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
            model=load_model(current_model_save_filepath_full,custom_objects=None, compile=True)
            for image_class in class_list:
                train_dir_base="/home/daniel/replay/imagenet_split/val"
                train_dir_full=os.path.join(train_dir_base,image_class)
                os.chdir(train_dir_full)
                list_of_files=os.listdir(train_dir_full)
                list_of_files.sort()
                for files in list_of_files:
                    os.chdir(train_dir_full)
                    img = image.load_img(files, target_size=(224, 224))
                    img_tensor = image.img_to_array(img)
                    img_tensor = np.expand_dims(img_tensor, axis=0)
                    x_train=preprocess_input(img_tensor)
                    test_activation=keract.get_activations(model, x_train, layer_names='block4_pool', nodes_to_evaluate=None, output_format='simple', auto_compile=True)
                    test_activation_values=test_activation.values()
                    data = list(test_activation_values)
                    data_array = np.array(data)
                    data_array_squeezed=np.squeeze(data_array)
                    data_array_squeezed=data_array_squeezed.astype('float16')
                    basedir='/home/daniel/replay/replay_training/val_activations'
                    activation_dir = os.path.join(basedir,image_class)
                    if not os.path.exists(activation_dir):
                        os.makedirs(activation_dir)
                    os.chdir(activation_dir)
                    extension='.npy'
                    data_title=files+extension
                    np.save(data_title, data_array_squeezed)
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
                    del model 
                except:
                    pass
                print(gc.collect()) 
                config = tensorflow.compat.v1.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 1
                config.gpu_options.visible_device_list = "0"
                set_session(tensorflow.compat.v1.Session(config=config))
            reset_keras()        
        val_activations(current_model_save_filepath_full)
        
        # use the validation set 
        def distributions():
            import numpy as np
            class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
            for image_class in class_list:
                activation_path_base='/home/daniel/replay/replay_training/train_augmented_activations'
                activation_path_full = os.path.join(activation_path_base,image_class)
                os.chdir(activation_path_full)
                list_of_files=os.listdir(activation_path_full)
                list_of_files.sort()
                h=0
                # rearrange all activations into a single variable
                for files in list_of_files:
                    if h==0:
                        temp_data=np.load(files)
                        temp_data=np.expand_dims(temp_data, axis=0)
                        h=h+1
                    else:
                        temp_data_2=np.load(files)
                        temp_data_2=np.expand_dims(temp_data_2, axis=0)
                        temp_data=np.append(temp_data, temp_data_2, axis=0)
                        h=h+1
                        print(h)
                # get the dimensions of the activations (height x width x filter)
                # for block 5, these will remain as-is for the calculating the 
                # multivariate distribution. For 3 and 4, height and width will
                # be downsampled by a factor of two, and for blocks 1 and 2,
                # will be downsampled by a factor of four due to RAM constraints
                len_x1=(len(np.ndarray.flatten((temp_data[0,:,0,0]))))/2
                len_x2=(len(np.ndarray.flatten((temp_data[0,0,:,0]))))/2
                len_x3=len(np.ndarray.flatten((temp_data[0,0,0,:])))
                len_x=len_x1*len_x2*len_x3
                len_y=len(list_of_files)
                # creates a new variable which is the flattened activation x 
                # number of activations/images
                all_data=np.zeros((int(len_x),int(len_y)))
                for images in range(len(list_of_files)):
                    new_filter=np.zeros((int(len_x1),int(len_x2),int(len_x3)))
                    for n in range(int(len_x3)):
                        temp_filter_full=temp_data[images,:,:,n]
                        temp_filter_full = np.reshape(temp_filter_full, (int(len_x1), 2, int(len_x1), 2))
                        temp_filter_downsampled = np.mean(np.mean(temp_filter_full, axis=3), axis=1)
                        new_filter[:,:,n]=temp_filter_downsampled
                    all_data[:,images]=np.ndarray.flatten(new_filter)
                # get the mean and covariance of this variable
                mean_data=np.mean(all_data,axis=1)
                covariance_matrix=np.cov(all_data)
                from scipy.stats import multivariate_normal
                # sample from a multivariate distribution
                test_sample = multivariate_normal.rvs(mean_data, covariance_matrix, size=1170)
                # for each of these activations, resample if necessary back to the original 
                # resolution, and save them as separate files for replay training
                for n in range(1170):
                    temp_sample=test_sample[n,:]
                    temp_sample_reshaped=np.reshape(temp_sample, (int(len_x1),int(len_x1), int(len_x3)))
                    new_upsampled_array=np.zeros((int(len_x1)*2,int(len_x1)*2, int(len_x3)))
                    for m in range(int(len_x3)):
                        temp_filter=temp_sample_reshaped[:,:,m]
                        temp_filter_upsampled=temp_filter.repeat(2, axis=0).repeat(2, axis=1)
                        new_upsampled_array[:,:,m]=temp_filter_upsampled[:,:] 
                    new_upsampled_array= new_upsampled_array.astype('float16')
                    upsampled_distribution_data_path_base='/home/daniel/replay/replay_training/train_augmented_distributions'
                    upsampled_distribution_data_path_full=os.path.join(upsampled_distribution_data_path_base,image_class)
                    if not os.path.exists(upsampled_distribution_data_path_full):
                        os.makedirs(upsampled_distribution_data_path_full)
                    os.chdir(upsampled_distribution_data_path_full)
                    sample_number=str(n)
                    sample_extension='.npy'
                    sample_filename=sample_number+sample_extension
                    np.save(sample_filename,new_upsampled_array)
        distributions()
        
        # replays the "imagined" representations through the end of the trained network
        def replay_vgg(current_model_save_filepath_full):
            # path to the replay images
            train_dir='/home/daniel/replay/replay_training/train_augmented_distributions'
            class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
            train_sets=[]
            for dp, dn, fn in os.walk(train_dir):
                dn[:] = [d for d in dn if d in class_list]
                for f in fn:    
                    train_sets.append((os.path.join(dp, f), dp.split('/')[-1]))
            x_train, y_cls_train = zip(*train_sets)
            folder_numbers=list(range(10))
            y_classes = {class_list[i]:folder_numbers[i] for i in range(10)}
            y_train = [y_classes[y] for y in y_cls_train]
            import tensorflow    
            y_train_oh = tensorflow.keras.utils.to_categorical(np.copy(y_train))
            # path to the validation activations from that point in the network
            validation_dir='/home/daniel/replay/replay_training/val_activations'
            val_sets=[]
            for dp, dn, fn in os.walk(validation_dir):
                dn[:] = [d for d in dn if d in class_list]
                for f in fn:    
                    val_sets.append((os.path.join(dp, f), dp.split('/')[-1]))
            x_val, y_cls_val = zip(*val_sets)
            y_val = [y_classes[y] for y in y_cls_val]
            y_val_oh = tensorflow.keras.utils.to_categorical(np.copy(y_val))
            type(np.asarray([0, 1, 2]))
            import math
            # log the training metrics for the replay
            checkpoint_directory_base = '/home/daniel/replay/replay_training/checkpoint'
            checkpoint_directory_full=os.path.join(checkpoint_directory_base,str(vgg_model))
            if not os.path.exists(checkpoint_directory_full):
                os.makedirs(checkpoint_directory_full)
            checkpoint_filename= current_model_save_filepath_full
            history_filename = os.path.join(checkpoint_directory_full, 'history_log.csv')
            batch_size_train=36
            batch_size_val=26
            import tensorflow.keras
            # build generator sequence
            class ImagenetSequence(tensorflow.keras.utils.Sequence):
                def __init__(self, x_set, y_set, batch_size):
                    self.x, self.y = x_set, y_set
                    self.batch_size = batch_size
                    self.shuffle_order = np.arange(len(self.x))
                    #np.random.seed(int(time.time()))
                    np.random.shuffle(self.shuffle_order)
                    self.x_shuf = self.x[self.shuffle_order]
                    self.y_shuf = self.y[self.shuffle_order]
                def __len__(self):
                    return math.ceil(len(self.x) / self.batch_size)
                def __getitem__(self, idx):
                    batch_x = self.x_shuf[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_y = self.y_shuf[idx * self.batch_size:(idx + 1) * self.batch_size]
                    return np.array([np.load(file_name) for file_name in batch_x]), np.array(batch_y)
                def on_epoch_end(self):
                    np.random.shuffle(self.shuffle_order)
            train_seq = ImagenetSequence(np.asarray(x_train),
                                         np.asarray(y_train_oh), 
                                         batch_size=batch_size_train)
            val_seq = ImagenetSequence(np.asarray(x_val),
                                       np.asarray(y_val_oh), 
                                       batch_size=batch_size_val)
            from tensorflow.keras.models import load_model
            model=load_model(current_model_save_filepath_full,custom_objects=None, compile=True)
            from tensorflow.keras import models
            from tensorflow.keras import layers
            from tensorflow.keras import optimizers
            # build new model with the input shape matching the replay representations
            # the rest of the replay model is taken from the current training model
            idx = 15  
            input_shape = model.layers[idx].get_input_shape_at(0) 
            input_shape2=[0,0,0]
            input_shape2[0]=input_shape[1]
            input_shape2[1]=input_shape[2]
            input_shape2[2]=input_shape[3]
            input_shape3=tuple(input_shape2)
            from tensorflow.keras.layers import Input
            layer_input = Input(shape=input_shape3) 
            x = layer_input
            for layer in model.layers[idx:]:
                x = layer(x)
            from tensorflow.keras.models import Model 
            new_model = Model(layer_input, x)                           
            new_model.summary()
            new_model.compile(optimizer=optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1.0, decay=0.0),
                        loss="categorical_crossentropy", 
                        metrics=["acc", "top_k_categorical_accuracy"])
            # the new replay model has lost its optimizer weights from the main model
            # this section reinitialises the replay model with optimizer weights, then
            # takes the optimizer weights from the relevant layers in the main model
            # and copies them into the replay model
            dummy_data=np.zeros((14,14,512))
            dummy_data=np.expand_dims(dummy_data,axis=0)
            dummy_labels=np.zeros((10))
            dummy_labels=np.expand_dims(dummy_labels,axis=0)
            new_model.fit(dummy_data,dummy_labels)
            full_model_symbolic_weights=getattr(model.optimizer, 'weights')
            from tensorflow.keras import backend as K
            full_model_optimizer_weight_values = K.batch_get_value(full_model_symbolic_weights)
            full_model_optimizer_weight_values=np.delete(full_model_optimizer_weight_values,(1,2,3,4,5,6,7,8,9,10,11,12,13,
                                                        14,15,16,17,18,19,20,33,34,35,36,
                                                        37,38,39,40,41,42,43,44,45,46,47,
                                                        48,49,50,51,52))
            new_model.optimizer.set_weights(full_model_optimizer_weight_values)
            from tensorflow.keras.callbacks import CSVLogger
            csv_logger = CSVLogger(history_filename, append=True, separator=';')
            step_size_train = len(train_sets) // batch_size_train
            step_size_val = len(val_sets) // batch_size_val
            print ("train step size:", step_size_train)
            print ("train step val:", step_size_val)
            # here we just train the replay model for one epoch
            epochs=1
            history = new_model.fit(train_seq,
                                    epochs=epochs,
                                    verbose=1,
                                    callbacks=[csv_logger],
                                    validation_data=val_seq,
                                    steps_per_epoch=step_size_train,
                                    validation_steps=step_size_val,
                                    max_queue_size=16,
                                    workers=8,
                                    use_multiprocessing=False
                                    )
            new_model_savepath=os.path.join(checkpoint_directory_full,'replay_model.h5')
            new_model.save(new_model_savepath)
            # we now need to copy the new model weights, and optimizer weights from the replay
            # model back into the main model
            # load the new weights back into the main model
            model.load_weights(new_model_savepath,by_name=True)
            # get the main model optimizer weights
            full_model_symbolic_weights=getattr(model.optimizer, 'weights')
            from tensorflow.keras import backend as K
            full_model_optimizer_weight_values = K.batch_get_value(full_model_symbolic_weights)
            # get the replay model optimizer weights
            replay_symbolic_weights = getattr(new_model.optimizer, 'weights')
            replay_optimizer_weight_values = K.batch_get_value(replay_symbolic_weights)
            untouched_full_weights=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                    15,16,17,18,19,20,33,34,35,36,
                    37,38,39,40,41,42,43,44,45,46,47,
                    48,49,50,51,52]
            count_replay=0
            for n in range(65):
                # for all the optimizer weights that were changed, copy them from the replay
                # model back into the main model
                if n not in untouched_full_weights:
                    full_model_optimizer_weight_values[n]=replay_optimizer_weight_values[count_replay]
                    print(count_replay)
                    print(full_model_optimizer_weight_values[n].shape)
                    print(replay_optimizer_weight_values[count_replay].shape)
                    count_replay=count_replay+1  
            model.optimizer.set_weights(full_model_optimizer_weight_values)
            # save the updated model after replay
            model.save(current_model_save_filepath_full)
            current_loss=history.history['val_loss'][:]
            current_loss_filepath=os.path.join(checkpoint_directory_full,'lowest_loss.npy')
            previous_lowest_loss=np.load(current_loss_filepath)
            # if the model improved after replay as measured by validation loss,
            # update it as the best model
            if current_loss<previous_lowest_loss:
                np.save(current_loss_filepath,current_loss)
                best_model_filepath=os.path.join(checkpoint_directory_full,'best_model.h5')
                model.save(best_model_filepath)
            # clear the gpu memory
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
                    del model 
                except:
                    pass
                try:
                    del new_model 
                except:
                    pass
                print(gc.collect()) 
                config = tensorflow.compat.v1.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 1
                config.gpu_options.visible_device_list = "0"
                set_session(tensorflow.compat.v1.Session(config=config))
            reset_keras()   
        replay_vgg(current_model_save_filepath_full)
        class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
        import shutil
        # delete all the temporary training files
        for image_class in class_list:
            train_dir_augmented_base  = '/home/daniel/replay/replay_training/train_augmented'
            train_dir_augmented_full = os.path.join(train_dir_augmented_base,image_class)
            activations_base_dir='/home/daniel/replay/replay_training/train_augmented_activations'
            activations_dir_full = os.path.join(activations_base_dir,image_class)
            distribution_data_path_base='/home/daniel/replay/replay_training/train_augmented_distributions'
            distribution_data_path_full=os.path.join(distribution_data_path_base,image_class)
            val_activations_base_dir='/home/daniel/replay/replay_training/val_activations'
            val_activations_dir_full = os.path.join(val_activations_base_dir,image_class)  
            shutil.rmtree(train_dir_augmented_full)
            shutil.rmtree(activations_dir_full)
            shutil.rmtree(distribution_data_path_full)
            shutil.rmtree(val_activations_dir_full)