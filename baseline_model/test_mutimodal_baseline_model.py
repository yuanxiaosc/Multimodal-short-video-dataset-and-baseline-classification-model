import tensorflow as tf
from mutimodal_baseline_model import create_text_baseline_model, create_image_baseline_model, \
    create_video_baseline_model, create_multimodal_baseline_model

if __name__ == '__main__':
    shuffle_data = True
    BATCH_SIZE = 5
    REPEAT_DATASET = None

    vocab_size = 15798 + 1  # 1 for unknown
    txt_maxlen = 20
    image_height = 270
    image_width = 480
    image_channels = 3
    max_video_frame_number = 100
    video_height = 360
    video_width = 640
    video_channels = 3

    label_number = 31

    batch_txt_data = tf.random.uniform((BATCH_SIZE, txt_maxlen), 0, vocab_size, dtype=tf.int32)
    print("batch_txt_data.shape", batch_txt_data.shape)
    text_model = create_text_baseline_model(txt_maxlen, vocab_size, embedding_dim=100, lstm_units=64, output_dim=50)
    tf.keras.utils.plot_model(text_model, show_shapes=True, to_file='text_model_baseline_model.png')
    batch_txt_feature = text_model(batch_txt_data)
    print("batch_txt_feature.shape", batch_txt_feature.shape)
    print("")

    batch_image_data = tf.random.normal(shape=(BATCH_SIZE, image_height, image_width, image_channels))
    print("batch_image_data.shape", batch_image_data.shape)
    image_model = create_image_baseline_model(image_height, image_width, image_channels, output_dim=50)
    tf.keras.utils.plot_model(image_model, show_shapes=True, to_file='image_model_baseline_model.png')
    batch_image_feature = image_model(batch_image_data)
    print("batch_image_feature.shape", batch_image_feature.shape)
    print("")

    batch_video_data = tf.random.normal(
        shape=(BATCH_SIZE, max_video_frame_number, video_height, video_width, video_channels))
    print("batch_video_data.shape", batch_video_data.shape)
    video_model = create_video_baseline_model(max_video_frame_number, video_height, video_width, video_channels,
                                              output_dim=50)
    tf.keras.utils.plot_model(video_model, show_shapes=True, to_file='video_model_baseline_model.png')
    batch_video_feature = video_model(batch_video_data)
    print("batch_video_feature.shape", batch_video_feature.shape)
    print("")

    batch_txt_data = tf.random.uniform((BATCH_SIZE, txt_maxlen), 0, vocab_size, dtype=tf.int32)
    batch_image_data = tf.random.normal(shape=(BATCH_SIZE, image_height, image_width, image_channels))
    batch_video_data = tf.random.normal(
        shape=(BATCH_SIZE, max_video_frame_number, video_height, video_width, video_channels))

    multimodal_model = create_multimodal_baseline_model(label_number=label_number, txt_maxlen=txt_maxlen,
                                                        text_vocab_size=vocab_size, text_embedding_dim=100,
                                                        text_lstm_units=64, text_output_dim=50,
                                                        image_height=image_height, image_width=image_width,
                                                        image_channels=image_channels, image_output_dim=50,
                                                        max_video_frame_number=max_video_frame_number,
                                                        video_height=video_height, video_width=video_width,
                                                        video_channels=video_channels, video_output_dim=50)

    tf.keras.utils.plot_model(multimodal_model, show_shapes=True, to_file='multimodal_baseline_model.png')
    multimodal_model_out = multimodal_model([batch_txt_data, batch_image_data, batch_video_data])
    print("multimodal_model_out.shape", multimodal_model_out.shape)
