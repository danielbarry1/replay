import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
for vgg_model in range(10):
    import numpy as np
    import os
    import tensorflow.keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    original_model_save_filepath_base='/home/daniel/replay/original_models/'+str(vgg_model)
    original_model_save_filepath_full=original_model_save_filepath_base+'/original_vgg.h5'
    from tensorflow.keras.applications import VGG16
    base_model = VGG16(weights=None,include_top=False,input_shape=(224, 224, 3))
    #load base of vgg16, which has been fine tuned for two epochs on regular I
    base_model.load_weights('/home/daniel/fine_tune/checkpoint/fine_tuned_vgg.h5',by_name=True)
    base_model.summary()
    num_classes = 10
    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = base_model.inputs, outputs = predictions)
    from tensorflow.keras import optimizers
    model.compile(optimizer=optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1.0, decay=0.0),
             loss="categorical_crossentropy", 
             metrics=["acc", "top_k_categorical_accuracy"])       
    model.summary()

