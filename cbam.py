from keras import backend
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Input, Dense, Activation, Add, Multiply, Concatenate, RepeatVector, Reshape, Lambda


def BAM(input_tensor, r=16, d=4):

    input_dimension_shape = input_tensor.shape # (?, 28, 28, 64)

    _h = int(input_dimension_shape[1])
    _w = int(input_dimension_shape[2])
    num_channels = int(input_dimension_shape[3])

    # channel attention
    gap = GlobalAveragePooling2D()(input_tensor) # (B, C)
    fc = Dense(int(num_channels/r))(gap)
    c_attention = Dense(num_channels)(fc)
    c_attention_bn = BatchNormalization()(c_attention) # (B, C)


    # spatial attention
    conv_1_1 = Conv2D(int(num_channels/r), 1, strides=1, padding="same", data_format='channels_last')(input_tensor)
    conv_3_3 = Conv2D(int(num_channels/r), 3, strides=1, padding="same", dilation_rate=d, data_format='channels_last')(conv_1_1)
    conv_3_3 = Conv2D(int(num_channels/r), 3, strides=1, padding="same", dilation_rate=d, data_format='channels_last')(conv_3_3)
    s_attention = Conv2D(1, 1, strides=1, padding="same", data_format='channels_last')(conv_3_3)
    s_attention_bn = BatchNormalization()(s_attention) # (B, H, W, 1)


    print("c_attention_bn", c_attention_bn)    # (?, 64)
    print("s_attention_bn", s_attention_bn)    # (?, 28, 28, 1)
    # projection
    c_att__w = RepeatVector(_h*_w)(c_attention_bn) # (B, W, C) # (?, 28, 64) # (?, 784, 64)
    print("c_att__w", c_att__w)

    c_att__h_w = Reshape([_h, _w, num_channels])(c_att__w)
    print("c_att__h_w", c_att__h_w)

    s_att__c = Lambda(lambda x:backend.repeat_elements(x, num_channels, 3))(s_attention_bn) # (B, H, W, 1*num_channels)
    print("s_att__c", s_att__c)

    _sum = Add()([c_att__h_w, s_att__c])
    bam = Activation('sigmoid')(_sum)
    _mul = Multiply()([input_tensor, bam])
    return _mul

def CBAM(input_tensor, r=16, name=None):

    input_dimension_shape = input_tensor.shape # (?, 28, 28, 64)

    # 2D CNN
    _h = int(input_dimension_shape[1])
    _w = int(input_dimension_shape[2])
    _c = int(input_dimension_shape[3])

    #img_input = Input(shape=(_h, _w, _c))

    def get_channel_attention_module(c_size, r, name='c_att_mod'):
        #input_dimension_shape = input_tensor.shape # (?, 28, 28, 64)
        #print("debug, get_channel_attention_module", input_dimension_shape)

        #_c = int(input_dimension_shape[1])
        _c_input = Input(shape=(c_size, )) #
        x = Activation('relu')(_c_input)
        x = Dense(int(_c/r))(x)
        x = Dense(c_size)(x)

        model = Model(_c_input, x, name=name)
        return model

    # channel attention
    c_gap = GlobalAveragePooling2D()(input_tensor) # (B, C)
    c_gmp = GlobalMaxPooling2D()(input_tensor)
    #print("debug, get_channel_attention_module", input_tensor) # # (?, 28, 28, 256)
    c_att_mod = get_channel_attention_module(int(c_gap.shape[1]), r, name=name+'_c_att_mod')

    _c_gap = c_att_mod(c_gap)
    _c_gmp = c_att_mod(c_gmp)

    x = Add()([_c_gap, _c_gmp])
    x = Activation('sigmoid')(x) # (B, C)
    x = RepeatVector(_h*_w)(x)
    #print("debug, RepeatVector, x", x) # (?, 784, 16)
    x = Reshape([_h, _w, _c])(x)
    f_ = Multiply()([input_tensor, x])

    # spatial attention
    s_ap = Lambda(lambda x:backend.mean(x, axis=-1, keepdims=True))(f_)
    s_mp = Lambda(lambda x:backend.max(x, axis=-1, keepdims=True))(f_)
    #print("debug, s_ap, x", s_ap) # (?, 28, 28, 1)
    #print("debug, s_mp, x", s_mp) # (?, 28, 28, 1)
    x = Concatenate(axis=-1)([s_ap, s_mp])
    #print("debug, Concatenate, x", x)
    x = Conv2D(_c, 7, strides=1, padding="same", data_format='channels_last')(x)
    x = Activation('sigmoid')(x)
    print("debug, Activation, x", x) # (?, 784, 16)
    f__ = Multiply()([f_, x])

    return f__
