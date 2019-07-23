import tensorflow as tf


def create_text_baseline_model(txt_max_len, vocab_size, embedding_dim=100, lstm_units=64, output_dim=50):
    text_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(txt_max_len), name='text_data'),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_dim, activation='relu')
    ], name='text_baseline_model')
    return text_model


def create_image_baseline_model(image_height, image_width, image_channels=3, output_dim=50):
    image_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(image_height, image_width, image_channels), name='image_data'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_dim)
    ], name='image_baseline_model')
    return image_model


def create_video_baseline_model(max_video_frame_number, video_height, video_width, video_channels=3, output_dim=50):
    """
    :param input_shape: [video sequences length, video_height, video_width, video_channels]
    :return: 3D_convolutional model
    """

    def get_con3d_block(filters=64, kernel_size=(3, 3, 3),
                        strides=(1, 1, 1), padding='same'):
        return tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides,
                                      padding=padding, data_format='channels_last',
                                      dilation_rate=(1, 1, 1), activation='relu',
                                      use_bias=True, kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros', kernel_regularizer=None,
                                      bias_regularizer=None, activity_regularizer=None,
                                      kernel_constraint=None, bias_constraint=None)

    def get_maxpooling3d_block(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same'):
        return tf.keras.layers.MaxPooling3D(pool_size=pool_size, strides=strides,
                                            padding=padding, data_format='channels_last')

    model = tf.keras.models.Sequential(name="video_baseline_model")
    # Input
    model.add(tf.keras.layers.Input(shape=(max_video_frame_number, video_height, video_width, video_channels),
                                    name='video_data'))

    # Conv3D + MaxPooling3D
    model.add(get_con3d_block(filters=32, kernel_size=(3, 3, 3)))
    model.add(get_maxpooling3d_block(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(get_con3d_block(filters=32, kernel_size=(3, 3, 3)))
    model.add(get_maxpooling3d_block(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(get_con3d_block(filters=64, kernel_size=(3, 3, 3)))
    model.add(get_maxpooling3d_block(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(get_con3d_block(filters=64, kernel_size=(3, 3, 3)))
    model.add(get_maxpooling3d_block(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(get_con3d_block(filters=128, kernel_size=(3, 3, 3)))
    model.add(get_maxpooling3d_block(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(get_con3d_block(filters=128, kernel_size=(3, 3, 3)))
    model.add(get_maxpooling3d_block(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(get_con3d_block(filters=256, kernel_size=(3, 3, 3)))
    model.add(get_maxpooling3d_block(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(get_con3d_block(filters=256, kernel_size=(3, 3, 3)))
    model.add(get_maxpooling3d_block(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    # Flatten
    model.add(tf.keras.layers.Flatten())

    # FC layers group
    model.add(tf.keras.layers.Dense(256, activation='relu', name='fc6'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu', name='fc7'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(output_dim))

    return model


def create_multimodal_baseline_model(label_number=31, txt_max_len=20, text_vocab_size=15799, text_embedding_dim=100,
                                     text_lstm_units=64, text_output_dim=50,
                                     image_height=270, image_width=480, image_channels=3, image_output_dim=50,
                                     max_video_frame_number=100, video_height=360, video_width=640, video_channels=3,
                                     video_output_dim=50):
    """
    Multimodal Baseline Model
    Text model parameters:
    [ vocab_size, txt_max_len, text_embedding_dim, text_lstm_units, text_output_dim]

    Image model parameters:
    [image_height, image_width, image_channels, image_output_dim]

    Video model parameters:
    [max_video_frame_number, video_height, video_width, video_channels, video_output_dim]

    label_number
    """
    text_input = tf.keras.layers.Input(shape=(txt_max_len), name='text_data')
    image_input = tf.keras.layers.Input(shape=(image_height, image_width, image_channels), name='image_data')
    video_input = tf.keras.layers.Input(shape=(max_video_frame_number, video_height, video_width, video_channels),
                                        name='video_data')

    text_model = create_text_baseline_model(txt_max_len, text_vocab_size, text_embedding_dim, text_lstm_units,
                                            text_output_dim)
    image_model = create_image_baseline_model(image_height, image_width, image_channels, image_output_dim)
    video_model = create_video_baseline_model(max_video_frame_number, video_height, video_width, video_channels,
                                              video_output_dim)

    text_feature = text_model(text_input)
    image_feature = image_model(image_input)
    video_feature = video_model(video_input)

    multimodal_feature = tf.keras.layers.concatenate([text_feature, image_feature, video_feature], axis=-1)

    x = tf.keras.layers.Dense(100)(multimodal_feature)
    label_predict = tf.keras.layers.Dense(label_number, activation='softmax', name='video_label')(x)

    multimodal_baseline_model = tf.keras.Model(inputs=[text_input, image_input, video_input], outputs=[label_predict])
    return multimodal_baseline_model
