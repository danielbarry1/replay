import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
for vgg_model in range(10):
    original_model_save_filepath_base='/home/daniel/replay/replay_training/original_models/'+str(vgg_model)
    original_model_save_filepath_full=original_model_save_filepath_base+'/original_vgg.h5'
    current_model_save_filepath_base='/home/daniel/replay/reinforcement_replay/checkpoint/'+str(vgg_model)
    import shutil
    if os.path.exists(current_model_save_filepath_base):
        shutil.rmtree(current_model_save_filepath_base)
    if not os.path.exists(current_model_save_filepath_base):
        os.makedirs(current_model_save_filepath_base)
    current_model_save_filepath_full=current_model_save_filepath_base+'/current_model.h5'
    class_labels_directory_base_temp = '/home/daniel/replay/reinforcement_replay/actual_training/'
    class_labels_directory_full_temp = os.path.join(class_labels_directory_base_temp,str(vgg_model))
    if os.path.exists(class_labels_directory_full_temp):
        shutil.rmtree(class_labels_directory_full_temp)
    os.makedirs(class_labels_directory_full_temp)
    for training_epoch in range(30):
        class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='reflect')
        for image_class in class_list:
            train_dir_augmented_base  = "/home/daniel/replay/reinforcement_replay/train_augmented/"
            train_dir_augmented_full = train_dir_augmented_base+image_class
            if not os.path.exists(train_dir_augmented_full):
                os.makedirs(train_dir_augmented_full)
            train_dir_base  = "/home/daniel/replay/imagenet_split/train/"
            train_dir_full = train_dir_base+image_class
            list_of_files=os.listdir(train_dir_full)
            list_of_files.sort()
            os.chdir(train_dir_full)
            for filename in list_of_files:
                temp_img=datagen.apply_transform(img_to_array(load_img(filename,target_size=(224, 224))),(datagen.get_random_transform((224,224,3))))
                temp_img=temp_img.astype(np.uint8)
                temp_img = Image.fromarray(temp_img)
                save_filename=os.path.join(train_dir_augmented_full,filename)
                temp_img.save(save_filename)
        def training(training_epoch,original_model_save_filepath_full,current_model_save_filepath_full):
            import numpy as np
            import os
            import tensorflow.keras
            from tensorflow.keras.callbacks import CSVLogger
            import numpy as np
            from tensorflow.keras.models import load_model
            import math
            if training_epoch==0:
                model=load_model(original_model_save_filepath_full,custom_objects=None,compile=True)
            else:
                model=load_model(current_model_save_filepath_full,custom_objects=None,compile=True)
            class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
            train_dir  = "/home/daniel/replay/reinforcement_replay/train_augmented"
            validation_dir = "/home/daniel/replay/imagenet_split/val"
            test_dir="/home/daniel/replay/imagenet_split/test"
            checkpoint_directory_base = '/home/daniel/replay/reinforcement_replay/checkpoint'
            checkpoint_directory_full=os.path.join(checkpoint_directory_base,str(vgg_model))
            if not os.path.exists(checkpoint_directory_full):
                os.makedirs(checkpoint_directory_full)
            checkpoint_filename= os.path.join(checkpoint_directory_full, 'current_model.h5')
            history_directory_base = checkpoint_directory_full
            history_filename = os.path.join(history_directory_base, 'history_log.csv')
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
            batch_size_train=36
            batch_size_val=26
            batch_size_test=25
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
            class ImagenetSequence_predict(tensorflow.keras.utils.Sequence):
                def __init__(self, x_set, y_set, batch_size):
                    self.x, self.y = x_set, y_set
                    self.batch_size = batch_size
                def __len__(self):
                    return math.ceil(len(self.x) / self.batch_size)
                def __getitem__(self, idx):
                    batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
                    return np.array([preprocess_input((img_to_array(load_img(file_name,target_size=(224, 224)))))
                            for file_name in batch_x]), np.array(batch_y)
            # for speed, the validation set labels are saved and retrieved each time to make the 
            # confusion matrix
            x_val_predict_filepath = '/home/daniel/replay/val_labels/'+str(vgg_model)+'/validation_paths.npy'
            y_val_predict_filepath = '/home/daniel/replay/val_labels/'+str(vgg_model)+'/validation_labels.npy'
            y_val_oh_predict_filepath = '/home/daniel/replay/val_labels/'+str(vgg_model)+'/oh_validation_labels.npy'
            x_val_predict=np.load(x_val_predict_filepath)
            y_val_predict=np.load(y_val_predict_filepath)
            y_val_oh_predict=np.load(y_val_oh_predict_filepath)
            train_seq = ImagenetSequence(np.asarray(x_train),
                                         np.asarray(y_train_oh),
                                         batch_size=batch_size_train)
            val_seq = ImagenetSequence(np.asarray(x_val),
                                       np.asarray(y_val_oh),
                                       batch_size=batch_size_val)
            val_seq_predict = ImagenetSequence_predict(np.asarray(x_val_predict),
                                       np.asarray(y_val_oh_predict),
                                       batch_size=batch_size_val)
            test_seq = ImagenetSequence_predict(np.asarray(x_test),
                                       np.asarray(y_test_oh),
                                       batch_size=batch_size_test)
            csv_logger = CSVLogger(history_filename, append=True, separator=';')
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
            # the following code computes the confusion matrix of the networks predictions,
            # at every training epoch, and then the chi square of the observed correct predictions
            # of the model, and the possible correct predictions by the model
            Y_pred=model.predict(val_seq_predict, batch_size=None, verbose=1, steps=None, callbacks=None, max_queue_size=16,
            workers=8, use_multiprocessing=False)
            # find the networks top prediction for each validation image
            y_pred = np.argmax(Y_pred, axis=1)
            y_pred.astype('float64')
            class_labels=y_val_predict
            from sklearn.metrics import classification_report, confusion_matrix
            # generate confusion matrix which is the predictions of the network
            # for every image
            confusion_matrix_1=confusion_matrix(class_labels, y_pred)
            f_obs=np.zeros(len(confusion_matrix_1))
            # find the number of correct predictions along the diagonal
            for i in range(len(confusion_matrix_1)):
                f_obs[i]=confusion_matrix_1[i,i]
            # find the total number of images for each class
            f_exp=np.sum(confusion_matrix_1,axis=1)
            import scipy
            print('obs',f_obs)
            print('exp',f_exp)
            # compute the chi square between the two, observed and expected
            [chisq,p]=scipy.stats.chisquare(f_obs,f_exp)
            chisq_filepath_base=os.path.join('/home/daniel/replay/reinforcement_replay/actual_training/',str(vgg_model))
            chisq_filepath = os.path.join(chisq_filepath_base,'chisq.npy')
            # save the current chi square, this will be updated after each epoch
            # of training and replay
            np.save(chisq_filepath,chisq)
            # get the performance on the testing set for the model
            test_metrics=model.evaluate(test_seq)
            def append_list_as_row(file_name, list_of_elem):
                # Open file in append mode
                with open(file_name, 'a+', newline='') as write_obj:
                    from csv import writer
                    csv_writer = writer(write_obj)
                    csv_writer.writerow(list_of_elem)
            test_metrics_filepath_base=os.path.join('/home/daniel/replay/reinforcement_replay/checkpoint/',str(vgg_model))
            test_metrics_filepath=os.path.join(test_metrics_filepath_base,'test_validation_metrics.csv')
            append_list_as_row(test_metrics_filepath,test_metrics)
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

        def activations(current_model_save_filepath_full):
            import numpy as np
            from tensorflow.keras.applications.vgg16 import preprocess_input
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.models import load_model
            import keract
            class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
            model=load_model(current_model_save_filepath_full,custom_objects=None, compile=True)
            for image_class in class_list:
                train_dir_base="/home/daniel/replay/reinforcement_replay/train_augmented"
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
                    basedir='/home/daniel/replay/reinforcement_replay/train_augmented_activations'
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
        activations(current_model_save_filepath_full)

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
                    basedir='/home/daniel/replay/reinforcement_replay/val_activations'
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

        def distributions():
            import numpy as np
            class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
            for image_class in class_list:
                activation_path_base='/home/daniel/replay/reinforcement_replay/train_augmented_activations'
                activation_path_full = os.path.join(activation_path_base,image_class)
                os.chdir(activation_path_full)
                list_of_files=os.listdir(activation_path_full)
                list_of_files.sort()
                h=0
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
                len_x1=(len(np.ndarray.flatten((temp_data[0,:,0,0]))))/2
                len_x2=(len(np.ndarray.flatten((temp_data[0,0,:,0]))))/2
                len_x3=len(np.ndarray.flatten((temp_data[0,0,0,:])))
                len_x=len_x1*len_x2*len_x3
                len_y=len(list_of_files)
                all_data=np.zeros((int(len_x),int(len_y)))
                for images in range(len(list_of_files)):
                    new_filter=np.zeros((int(len_x1),int(len_x2),int(len_x3)))
                    for n in range(int(len_x3)):
                        temp_filter_full=temp_data[images,:,:,n]
                        temp_filter_full = np.reshape(temp_filter_full, (int(len_x1), 2, int(len_x1), 2))
                        temp_filter_downsampled = np.mean(np.mean(temp_filter_full, axis=3), axis=1)
                        new_filter[:,:,n]=temp_filter_downsampled
                    all_data[:,images]=np.ndarray.flatten(new_filter)
                mean_data=np.mean(all_data,axis=1)
                covariance_matrix=np.cov(all_data)
                from scipy.stats import multivariate_normal
                test_sample = multivariate_normal.rvs(mean_data, covariance_matrix, size=1170)
                for n in range(1170):
                    temp_sample=test_sample[n,:]
                    temp_sample_reshaped=np.reshape(temp_sample, (int(len_x1),int(len_x1), int(len_x3)))
                    new_upsampled_array=np.zeros((int(len_x1)*2,int(len_x1)*2, int(len_x3)))
                    for m in range(int(len_x3)):
                        temp_filter=temp_sample_reshaped[:,:,m]
                        temp_filter_upsampled=temp_filter.repeat(2, axis=0).repeat(2, axis=1)
                        new_upsampled_array[:,:,m]=temp_filter_upsampled[:,:]
                    new_upsampled_array= new_upsampled_array.astype('float16')
                    upsampled_distribution_data_path_base='/home/daniel/replay/reinforcement_replay/train_augmented_distributions'
                    upsampled_distribution_data_path_full=os.path.join(upsampled_distribution_data_path_base,image_class)
                    if not os.path.exists(upsampled_distribution_data_path_full):
                        os.makedirs(upsampled_distribution_data_path_full)
                    os.chdir(upsampled_distribution_data_path_full)
                    sample_number=str(n)
                    sample_extension='.npy'
                    sample_filename=sample_number+sample_extension
                    np.save(sample_filename,new_upsampled_array)
        distributions()

        # replay will now be called one batch at a time, where each batch is one class

        def replay_vgg(image_class,current_model_save_filepath_full,replay):
            class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
            training_path='/home/daniel/replay/reinforcement_replay/train_augmented_distributions'
            train_sets=[]
            include = image_class
            for dp, dn, fn in os.walk(training_path):
                dn[:] = [d for d in dn if d in include]
                for f in fn:
                    train_sets.append((os.path.join(dp, f), dp.split('/')[-1]))
            x_train, y_cls_train = zip(*train_sets)
            y_train_oh = np.zeros((len(train_sets),10))
            class_id=np.where(class_list == image_class)
            class_id=class_id[0][0]
            y_train_oh[:,class_id] = 1
            type(np.asarray([0, 1, 2]))
            import math
            batch_size_train=36
            batch_size_val=26
            import tensorflow.keras
            class ImagenetSequence_train(tensorflow.keras.utils.Sequence):
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
                    print('batch x',batch_x)
                    print('batch y',batch_y)
                    return np.array([np.load(file_name) for file_name in batch_x]), np.array(batch_y)
            class ImagenetSequence_predict(tensorflow.keras.utils.Sequence):
                def __init__(self, x_set, y_set, batch_size):
                    self.x, self.y = x_set, y_set
                    self.batch_size = batch_size
                def __len__(self):
                    return math.ceil(len(self.x) / self.batch_size)
                def __getitem__(self, idx):
                    batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
                    return np.array([np.load(file_name) for file_name in batch_x]), np.array(batch_y)
            x_val_predict_filepath = '/home/daniel/replay/reinforcement_replay/activation_val_labels/'+str(vgg_model)+'/validation_paths.npy'
            y_val_predict_filepath = '/home/daniel/replay/reinforcement_replay/activation_val_labels/'+str(vgg_model)+'/validation_labels.npy'
            y_val_oh_predict_filepath = '/home/daniel/replay/reinforcement_replay/activation_val_labels/'+str(vgg_model)+'/oh_validation_labels.npy'
            x_val_predict=np.load(x_val_predict_filepath)
            y_val_predict=np.load(y_val_predict_filepath)
            y_val_oh_predict=np.load(y_val_oh_predict_filepath)
            train_seq = ImagenetSequence_train(np.asarray(x_train),
                                         np.asarray(y_train_oh),
                                         batch_size=batch_size_train)
            val_seq_predict = ImagenetSequence_predict(np.asarray(x_val_predict),
                                       np.asarray(y_val_oh_predict),
                                       batch_size=batch_size_val)
            from tensorflow.keras.models import load_model
            # if this is the first batch of replay within a replay epoch, build the replay model
            # otherwise, just load the updated one from the disk
            if replay==0:
                model=load_model(current_model_save_filepath_full, custom_objects=None, compile=True)
                from tensorflow.keras import models
                from tensorflow.keras import layers
                from tensorflow.keras import optimizers
                idx = 15
                input_shape = model.layers[idx].get_input_shape_at(0) # get the input shape of desired layer
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
                replay_model = Model(layer_input, x)
                replay_model.summary()
                replay_model.compile(optimizer=optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1.0, decay=0.0),
                            loss="categorical_crossentropy",
                            metrics=["acc", "top_k_categorical_accuracy"])
                dummy_data=np.zeros((14,14,512))
                dummy_data=np.expand_dims(dummy_data,axis=0)
                dummy_labels=np.zeros((10))
                dummy_labels=np.expand_dims(dummy_labels,axis=0)
                replay_model.fit(dummy_data,dummy_labels)
                full_model_symbolic_weights=getattr(model.optimizer, 'weights')
                from tensorflow.keras import backend as K
                full_model_optimizer_weight_values = K.batch_get_value(full_model_symbolic_weights)
                full_model_optimizer_weight_values=np.delete(full_model_optimizer_weight_values,(1,2,3,4,5,6,7,8,9,10,11,12,13,
                                                            14,15,16,17,18,19,20,33,34,35,36,
                                                            37,38,39,40,41,42,43,44,45,46,47,
                                                            48,49,50,51,52))
                replay_model.optimizer.set_weights(full_model_optimizer_weight_values)
            else:
                 replay_model_directory_base = '/home/daniel/replay/reinforcement_replay/actual_training/'
                 replay_model_directory_full = os.path.join(replay_model_directory_base,str(vgg_model))
                 replay_model_filepath=os.path.join(replay_model_directory_full,'replay_model.h5')
                 replay_model=load_model(replay_model_filepath, custom_objects=None, compile=True)
            epochs=1
            history=replay_model.fit(train_seq,
                                steps_per_epoch=1,
                                epochs=epochs,
                                validation_data=None,
                                verbose=1,
                                )
            running_loss=history.history['loss'][:]
            running_acc=history.history['acc'][:]
            running_top_5=history.history['top_k_categorical_accuracy'][:]
            replay_model_directory_base = '/home/daniel/replay/reinforcement_replay/actual_training/'
            replay_model_directory_full = os.path.join(replay_model_directory_base,str(vgg_model))
            running_loss_filepath=os.path.join(replay_model_directory_full,'running_loss.npy')
            running_acc_filepath=os.path.join(replay_model_directory_full,'running_acc.npy')
            running_top_5_filepath=os.path.join(replay_model_directory_full,'running_top_5.npy')
            if replay==0:
                np.save(running_loss_filepath,running_loss)
                np.save(running_acc_filepath,running_acc)
                np.save(running_top_5_filepath,running_top_5)
            else:
                previous_running_loss=np.load(running_loss_filepath)
                previous_running_acc=np.load(running_acc_filepath)
                previous_running_top_5=np.load(running_top_5_filepath)
                updated_running_loss=np.append(previous_running_loss,running_loss)
                updated_running_acc=np.append(previous_running_acc,running_acc)
                updated_running_top_5=np.append(previous_running_top_5,running_top_5)
                np.save(running_loss_filepath,updated_running_loss)
                np.save(running_acc_filepath,updated_running_acc)
                np.save(running_top_5_filepath,updated_running_top_5)
            print('replay_training_history:',history.history)
            if replay==324:
                replay_metrics=np.zeros((3))
                replay_metrics[0]=np.mean(updated_running_loss)
                replay_metrics[1]=np.mean(updated_running_acc)
                replay_metrics[2]=np.mean(updated_running_top_5)
                def append_list_as_row(file_name, list_of_elem):
                    with open(file_name, 'a+', newline='') as write_obj:
                        from csv import writer
                        csv_writer = writer(write_obj)
                        csv_writer.writerow(list_of_elem)
                replay_training_metrics_filepath=os.path.join(replay_model_directory_full,'replay_training_metrics.csv')
                append_list_as_row(replay_training_metrics_filepath,replay_metrics)
                os.remove(running_loss_filepath)
                os.remove(running_acc_filepath)
                os.remove(running_top_5_filepath)
            replay_model_directory_base = '/home/daniel/replay/reinforcement_replay/actual_training/'
            replay_model_directory_full = os.path.join(replay_model_directory_base,str(vgg_model))
            replay_model_filepath=os.path.join(replay_model_directory_full,'replay_model.h5')
            replay_model.save(replay_model_filepath)
            if replay==324:
                current_model=load_model(current_model_save_filepath_full, custom_objects=None, compile=True)
                current_model.load_weights(replay_model_filepath,by_name=True)
                full_model_symbolic_weights=getattr(current_model.optimizer, 'weights')
                from tensorflow.keras import backend as K
                full_model_optimizer_weight_values = K.batch_get_value(full_model_symbolic_weights)
                replay_symbolic_weights = getattr(replay_model.optimizer, 'weights')
                replay_optimizer_weight_values = K.batch_get_value(replay_symbolic_weights)
                untouched_full_weights=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                        15,16,17,18,19,20,33,34,35,36,
                        37,38,39,40,41,42,43,44,45,46,47,
                        48,49,50,51,52]
                count_replay=0
                for n in range(65):
                    if n not in untouched_full_weights:
                        full_model_optimizer_weight_values[n]=replay_optimizer_weight_values[count_replay]
                        print(count_replay)
                        print(full_model_optimizer_weight_values[n].shape)
                        print(replay_optimizer_weight_values[count_replay].shape)
                        count_replay=count_replay+1
                current_model.optimizer.set_weights(full_model_optimizer_weight_values)
                current_model.save(current_model_save_filepath_full)
            # calculate the chi square value for the replay training each time
            Y_pred=replay_model.predict(val_seq_predict, batch_size=None, verbose=1, steps=None, callbacks=None, max_queue_size=16,
            workers=8, use_multiprocessing=False)
            y_pred = np.argmax(Y_pred, axis=1)
            y_pred.astype('float64')
            class_labels=y_val_predict
            count=0
            print(len(y_pred))
            print(len(class_labels))
            for n in range(len(class_labels)):
                 if class_labels[n]==y_pred[n]:
                         count=count+1
            from sklearn.metrics import classification_report, confusion_matrix
            confusion_matrix_1=confusion_matrix(class_labels, y_pred)
            f_obs=np.zeros(len(confusion_matrix_1))
            for i in range(len(confusion_matrix_1)):
                f_obs[i]=confusion_matrix_1[i,i]
            f_exp=np.sum(confusion_matrix_1,axis=1)
            import scipy
            print('obs',f_obs)
            print('exp',f_exp)
            [chisq,p]=scipy.stats.chisquare(f_obs,f_exp)
            def append_list_as_row(file_name, list_of_elem):
                with open(file_name, 'a+', newline='') as write_obj:
                    from csv import writer
                    csv_writer = writer(write_obj)
                    csv_writer.writerow(list_of_elem)
            chisquare_table=np.zeros(3)
            class_number=np.where(class_list == image_class)
            class_number=class_number[0]
            chisquare_table[0]=class_number
            chisquare_table[1]=chisq
            chisquare_table[2]=p
            chisq_filepath_base=os.path.join('/home/daniel/replay/reinforcement_replay/actual_training/',str(vgg_model))
            chisq_csv_filepath=os.path.join(chisq_filepath_base,'chisq.csv')
            append_list_as_row(chisq_csv_filepath,chisquare_table)
            ## this chisq comes from the previous training epoch, will get overwritten with every training epoch.
            chisq_filepath_base=os.path.join('/home/daniel/replay/reinforcement_replay/actual_training/',str(vgg_model))
            chisq_filepath = os.path.join(chisq_filepath_base,'chisq.npy')
            original_chisq=np.load(chisq_filepath)
            # calculate the change in chi square from replaying that class
            chisq_change=original_chisq-chisq
            np.save(chisq_filepath,chisq)
            # if it is the final replay batch, check if the model has improved on validation loss
            # and save model as best model if so
            if replay==324:
                history=replay_model.evaluate(val_seq_predict)
                current_loss=history[0]
                checkpoint_directory_full='/home/daniel/replay/reinforcement_replay/checkpoint/'+str(vgg_model)
                current_loss_filepath=os.path.join(checkpoint_directory_full,'lowest_loss.npy')
                previous_lowest_loss=np.load(current_loss_filepath)
                if current_loss<previous_lowest_loss:
                    np.save(current_loss_filepath,current_loss)
                    best_model_filepath=os.path.join(checkpoint_directory_full,'best_model.h5')
                    current_model.save(best_model_filepath)
                def append_list_as_row(file_name, list_of_elem):
                    with open(file_name, 'a+', newline='') as write_obj:
                        from csv import writer
                        csv_writer = writer(write_obj)
                        csv_writer.writerow(list_of_elem)
                replay_metrics_filepath_base=os.path.join('/home/daniel/replay/reinforcement_replay/actual_training/',str(vgg_model))
                replay_metrics_filepath=os.path.join(replay_metrics_filepath_base,'replay_validation_metrics.csv')
                append_list_as_row(replay_metrics_filepath,history)
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
                    del replay_model
                except:
                    pass
                print(gc.collect()) 
                config = tensorflow.compat.v1.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 1
                config.gpu_options.visible_device_list = "0"
                set_session(tensorflow.compat.v1.Session(config=config))
            reset_keras()
            return chisq_change
        
        # the reinforcement learning network is pre-trained to obtain reasonable
        # values to begin interacting with the main network.
        def get_initial_q_values(vgg_model,current_model_save_filepath_full):
            class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
            chisq_change=np.zeros(10)
            chisq_id=0
            for image_class in class_list:
                # assemble a list of paths to the replay representations
                training_path='/home/daniel/replay/reinforcement_replay/train_augmented_distributions'
                train_sets=[]
                include = image_class
                for dp, dn, fn in os.walk(training_path):
                    dn[:] = [d for d in dn if d in include]
                    for f in fn:
                        train_sets.append((os.path.join(dp, f), dp.split('/')[-1]))
                x_train, y_cls_train = zip(*train_sets)
                y_train_oh = np.zeros((len(train_sets),10))
                class_id=np.where(class_list == image_class)
                class_id=class_id[0][0]
                # image labels are for one class at a time
                y_train_oh[:,class_id] = 1
                type(np.asarray([0, 1, 2]))
                import math
                batch_size_train=36
                batch_size_val=26
                import tensorflow.keras
                class ImagenetSequence_train(tensorflow.keras.utils.Sequence):
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
                        return np.array([np.load(file_name) for file_name in batch_x]), np.array(batch_y)
                class ImagenetSequence_predict(tensorflow.keras.utils.Sequence):
                    def __init__(self, x_set, y_set, batch_size):
                        self.x, self.y = x_set, y_set
                        self.batch_size = batch_size
                    def __len__(self):
                        return math.ceil(len(self.x) / self.batch_size)
                    def __getitem__(self, idx):
                        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
                        return np.array([np.load(file_name) for file_name in batch_x]), np.array(batch_y)
                # for speed, the validation set labels are saved and retrieved each time to make the 
                # confusion matrix
                x_val_predict_filepath = '/home/daniel/replay/reinforcement_replay/activation_val_labels/'+str(vgg_model)+'/validation_paths.npy'
                y_val_predict_filepath = '/home/daniel/replay/reinforcement_replay/activation_val_labels/'+str(vgg_model)+'/validation_labels.npy'
                y_val_oh_predict_filepath = '/home/daniel/replay/reinforcement_replay/activation_val_labels/'+str(vgg_model)+'/oh_validation_labels.npy'
                x_val_predict=np.load(x_val_predict_filepath)
                y_val_predict=np.load(y_val_predict_filepath)
                y_val_oh_predict=np.load(y_val_oh_predict_filepath)
                train_seq = ImagenetSequence_train(np.asarray(x_train),
                                             np.asarray(y_train_oh),
                                             batch_size=batch_size_train)
                val_seq_predict = ImagenetSequence_predict(np.asarray(x_val_predict),
                                           np.asarray(y_val_oh_predict),
                                           batch_size=batch_size_val)
                from tensorflow.keras.models import load_model  
                model=load_model(current_model_save_filepath_full, custom_objects=None, compile=True)
                from tensorflow.keras import models
                from tensorflow.keras import layers
                from tensorflow.keras import optimizers
                # create replay model by using the end of the main model
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
                replay_model = Model(layer_input, x)
                replay_model.summary()
                replay_model.compile(optimizer=optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1.0, decay=0.0),
                            loss="categorical_crossentropy",
                            metrics=["acc", "top_k_categorical_accuracy"])
                dummy_data=np.zeros((14,14,512))
                dummy_data=np.expand_dims(dummy_data,axis=0)
                dummy_labels=np.zeros((10))
                dummy_labels=np.expand_dims(dummy_labels,axis=0)
                replay_model.fit(dummy_data,dummy_labels)
                full_model_symbolic_weights=getattr(model.optimizer, 'weights')
                from tensorflow.keras import backend as K
                # load the main model optimizer weights
                full_model_optimizer_weight_values = K.batch_get_value(full_model_symbolic_weights)
                full_model_optimizer_weight_values=np.delete(full_model_optimizer_weight_values,(1,2,3,4,5,6,7,8,9,10,11,12,13,
                                                            14,15,16,17,18,19,20,33,34,35,36,
                                                            37,38,39,40,41,42,43,44,45,46,47,
                                                            48,49,50,51,52))
                replay_model.optimizer.set_weights(full_model_optimizer_weight_values)
                epochs=1
                # run one batch of each class through the model
                history=replay_model.fit(train_seq,
                                    steps_per_epoch=1,
                                    epochs=epochs,
                                    validation_data=None,
                                    verbose=1,
                                    )
                print('replay_training_history:',history.history)
                Y_pred=replay_model.predict(val_seq_predict, batch_size=None, verbose=1, steps=None, callbacks=None, max_queue_size=16,
                workers=8, use_multiprocessing=False)
                y_pred = np.argmax(Y_pred, axis=1)
                y_pred.astype('float64')
                class_labels=y_val_predict
                count=0
                print(len(y_pred))
                print(len(class_labels))
                for n in range(len(class_labels)):
                     if class_labels[n]==y_pred[n]:
                             count=count+1
                from sklearn.metrics import classification_report, confusion_matrix
                confusion_matrix_1=confusion_matrix(class_labels, y_pred)
                f_obs=np.zeros(len(confusion_matrix_1))
                for i in range(len(confusion_matrix_1)):
                    f_obs[i]=confusion_matrix_1[i,i]
                f_exp=np.sum(confusion_matrix_1,axis=1)
                import scipy
                print('obs',f_obs)
                print('exp',f_exp)
                [chisq,p]=scipy.stats.chisquare(f_obs,f_exp)
                chisq_filepath_base=os.path.join('/home/daniel/replay/reinforcement_replay/actual_training/',str(vgg_model))
                chisq_filepath = os.path.join(chisq_filepath_base,'chisq.npy')
                original_chisq=np.load(chisq_filepath)
                # record the change in chi square for each class separately
                chisq_change[chisq_id]=original_chisq-chisq
                chisq_id=chisq_id+1
                # clear the gpu memory
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
                        del replay_model 
                    except:
                        pass
                    print(gc.collect()) 
                    config = tensorflow.compat.v1.ConfigProto()
                    config.gpu_options.per_process_gpu_memory_fraction = 1
                    config.gpu_options.visible_device_list = "0"
                    set_session(tensorflow.compat.v1.Session(config=config))
                reset_keras()
            return chisq_change
        # start the training
        if training_epoch==0:
            # run the initial pass to get values for the RL network to process
            chisq_change=get_initial_q_values(vgg_model,current_model_save_filepath_full)
            # create input labels which will be 10 separate inputs, and a value of 1 for each class
            input_labels=np.arange(10)
            import tensorflow.keras
            input_labels_oh = tensorflow.keras.utils.to_categorical(np.copy(input_labels))
            input_labels_oh_2=input_labels_oh
            from tensorflow.keras import models
            from tensorflow.keras import layers
            # create the RL network
            q_model=models.Sequential()
            # 10 inputs, one output
            q_model.add(layers.Dense(1,input_shape=(input_labels_oh_2.shape[1],)))
            q_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            # train the RL network on the 10 chi square changes, one for each class
            # done for 50 epochs
            q_model.fit(input_labels_oh_2,chisq_change,epochs=50,batch_size=10,verbose=1)
            # save the weights of the trained network
            weights=q_model.trainable_weights[0]
            weights_np=weights.numpy()
            actual_training_directory_base = '/home/daniel/replay/reinforcement_replay/actual_training/'
            actual_training_directory_full = os.path.join(actual_training_directory_base,str(vgg_model))
            if not os.path.exists(actual_training_directory_full):
                os.makedirs(actual_training_directory_full)
            os.chdir(actual_training_directory_full)
            np.savetxt('initial_weights.csv',weights_np)
            np.save('weights.npy',weights_np)
            # create a softmax layer from the weights layer.
            softmax_layer=tensorflow.nn.softmax(weights,axis=0).numpy()
            print(softmax_layer)
            np.savetxt('initial_softmax_layer.csv',softmax_layer)
            np.save('softmax_layer.npy',softmax_layer)
            q_model.save('q_model.h5')
        # load the current RL model for every epoch
        q_model_directory_base = '/home/daniel/replay/reinforcement_replay/actual_training/'
        q_model_directory_full = os.path.join(q_model_directory_base,str(vgg_model))
        q_model_filepath = os.path.join(q_model_directory_full,'q_model.h5')
        from tensorflow.keras.models import load_model
        q_model=load_model(q_model_filepath, custom_objects=None, compile=True)
        import numpy as np
        softmax_directory_base = '/home/daniel/replay/reinforcement_replay/actual_training/'
        softmax_directory_full = os.path.join(softmax_directory_base,str(vgg_model))
        softmax_filepath = os.path.join(softmax_directory_full,'softmax_layer.npy')
        # load the saved softmax layer
        softmax_layer=np.load(softmax_filepath)
        softmax_layer_flattened=softmax_layer.flatten()
        print('softmax layer',softmax_layer_flattened)
        # sample from the softmax layer with temperature
        # this makes the sampling from higher values more aggressive
        temperature=0.3
        preds=np.log(softmax_layer_flattened)/temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        # sample from the transformed softmax layer
        probas = np.random.multinomial(1, preds, 1)
        # choose the class to be replayed
        input_labels_oh=probas[0]
        input_labels_oh=np.asarray(input_labels_oh)
        input_labels_oh=input_labels_oh.astype('float32')
        class_number=np.where(input_labels_oh==1)
        print('decided to replay class:',class_number)
        class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
        image_class=class_list[class_number]
        input_labels_oh = np.expand_dims(input_labels_oh, axis=0)
        # now start the actual replay, beginning with that class
        for replay in range(325):
            # replay that chosen class through the main network
            chisq_change=replay_vgg(image_class,current_model_save_filepath_full,replay)
            chisq_change=np.asarray(chisq_change)
            chisq_change=chisq_change.astype('float32')
            chisq_change = np.expand_dims(chisq_change, axis=0)
            # train the RL model with the feedback from the main network
            q_model.fit(input_labels_oh,chisq_change,epochs=50,batch_size=1,verbose=1)
            # extract the new weights after training each class
            weights=q_model.trainable_weights[0]
            import tensorflow
            # sample aggressively from the softmax layer for the next trial
            softmax_layer=tensorflow.nn.softmax(weights,axis=0).numpy()
            np.save(softmax_filepath,softmax_layer)
            softmax_layer_flattened=softmax_layer.flatten()
            print('new softmax layer',softmax_layer_flattened)
            temperature=0.3
            preds=np.log(softmax_layer_flattened)/temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            input_labels_oh=probas[0]
            input_labels_oh=np.asarray(input_labels_oh)
            input_labels_oh=input_labels_oh.astype('float32')
            class_number=np.where(input_labels_oh==1)
            print('decided to replay class:',class_number)
            image_class=class_list[class_number]
            input_labels_oh = np.expand_dims(input_labels_oh, axis=0)
            # on the very last batch in the replay epoch, save the weights and softmax layer
            if replay==324:
                def append_list_as_row(file_name, list_of_elem):
                    with open(file_name, 'a+', newline='') as write_obj:
                        from csv import writer
                        csv_writer = writer(write_obj)
                        csv_writer.writerow(list_of_elem)
                actual_training_directory_base = '/home/daniel/replay/reinforcement_replay/actual_training/'
                actual_training_directory_full = os.path.join(actual_training_directory_base,str(vgg_model))
                if not os.path.exists(actual_training_directory_full):
                    os.makedirs(actual_training_directory_full)
                weights_filepath=os.path.join(actual_training_directory_full,'weights.csv')
                softmax_filepath=os.path.join(actual_training_directory_full,'softmax.csv')
                weights_transposed=weights.numpy()
                weights_transposed=np.transpose(weights_transposed)
                weights_transposed=weights_transposed[0]
                print('new weights',weights_transposed)
                append_list_as_row(weights_filepath,weights_transposed)
                append_list_as_row(softmax_filepath,softmax_layer_flattened)
        class_list=np.load('/home/daniel/replay/class_lists/'+str(vgg_model)+'.npy')
        import shutil
        # delete all the temporary files
        for image_class in class_list:
            train_dir_augmented_base  = "/home/daniel/replay/reinforcement_replay/train_augmented"
            train_dir_augmented_full = os.path.join(train_dir_augmented_base,image_class)
            activations_base_dir='/home/daniel/replay/reinforcement_replay/train_augmented_activations'
            activations_dir_full = os.path.join(activations_base_dir,image_class)
            distribution_data_path_base='/home/daniel/replay/reinforcement_replay/train_augmented_distributions'
            distribution_data_path_full=os.path.join(distribution_data_path_base,image_class)
            val_activations_base_dir='/home/daniel/replay/reinforcement_replay/val_activations'
            val_activations_dir_full = os.path.join(val_activations_base_dir,image_class)
            shutil.rmtree(train_dir_augmented_full)
            shutil.rmtree(activations_dir_full)
            shutil.rmtree(distribution_data_path_full)
            shutil.rmtree(val_activations_dir_full)