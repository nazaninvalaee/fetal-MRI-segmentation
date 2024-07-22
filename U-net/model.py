import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Activation
from tensorflow.keras.optimizers import Adam

def conv_block(input_tensor, num_filters):
    x = Conv3D(num_filters, (3, 3, 3), activation='relu', padding='same')(input_tensor)
    x = Conv3D(num_filters, (3, 3, 3), activation='relu', padding='same')(x)
    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling3D((2, 2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    x = UpSampling3D((2, 2, 2))(input_tensor)
    x = concatenate([x, concat_tensor])
    x = conv_block(x, num_filters)
    return x

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)
    
    b1 = conv_block(p4, 512)
    
    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)
    
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(d4)
    
    model = Model(inputs, outputs, name="3D-UNet")
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
