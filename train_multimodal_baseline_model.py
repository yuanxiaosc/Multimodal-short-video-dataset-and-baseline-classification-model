import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "data_interface_for_model")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "baseline_model")))
from baseline_model.mutimodal_baseline_model import create_multimodal_baseline_model
from data_interface_for_model.tensorflow_dataset_interface import multimodal_tensorflow_dataset


def train_multimodal_model_main(data_root, train_dataset_numbers, EPOCHS, LEARN_RATE,
                                checkpoint_path, shuffle_data, BATCH_SIZE, REPEAT_DATASET,
                                vocab_size, txt_maxlen,
                                image_height, image_width, image_channels,
                                max_video_frame_number, video_height, video_width, video_channels,
                                label_number):
    """
    Training Multimodal Baseline Model

    Control training and data parameters:
    [data_root, train_dataset_numbers, EPOCHS, LEARN_RATE,
    checkpoint_path, shuffle_data, BATCH_SIZE, REPEAT_DATASET,]

    Text model parameters:
    [ vocab_size, txt_maxlen,]

    Image model parameters:
    [image_height, image_width, image_channels,]

    Video model parameters:
    [max_video_frame_number, video_height, video_width, video_channels,]

    label_number
    """

    multimodal_dataset = multimodal_tensorflow_dataset(data_root, shuffle_data, BATCH_SIZE, REPEAT_DATASET,
                                                       txt_maxlen, image_height, image_width,
                                                       max_video_frame_number, video_height, video_width)

    # Create callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,
                                                             verbose=1, save_freq='epoch')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=checkpoint_path)
    callback_list = [checkpoint_callback, tensorboard_callback]

    STEPS_PER_EPOCH = train_dataset_numbers // BATCH_SIZE

    multimodal_model = create_multimodal_baseline_model(label_number=label_number, txt_maxlen=txt_maxlen,
                                                        text_vocab_size=vocab_size, text_embedding_dim=100,
                                                        text_lstm_units=64, text_output_dim=50,
                                                        image_height=image_height, image_width=image_width,
                                                        image_channels=image_channels, image_output_dim=50,
                                                        max_video_frame_number=max_video_frame_number,
                                                        video_height=video_height, video_width=video_width,
                                                        video_channels=video_channels, video_output_dim=50)

    multimodal_model.compile(optimizer=tf.keras.optimizers.Adam(LEARN_RATE),
                             loss=tf.keras.losses.CategoricalCrossentropy(),
                             metrics=[tf.keras.metrics.CategoricalAccuracy()])

    multimodal_model.fit(multimodal_dataset, epochs=EPOCHS,
                         steps_per_epoch=STEPS_PER_EPOCH, callbacks=callback_list)


if __name__ == "__main__":
    data_root = "/home/b418a/disk1/jupyter_workspace/yuanxiao/douyin/xinpianchang/MP4_download"
    train_dataset_numbers = 500000
    EPOCHS = 200
    LEARN_RATE = 0.001
    checkpoint_path = "./keras_checkpoints/train"

    shuffle_data = True
    BATCH_SIZE = 64
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

    train_multimodal_model_main(data_root, train_dataset_numbers, EPOCHS, LEARN_RATE, checkpoint_path,
                                shuffle_data, BATCH_SIZE, REPEAT_DATASET, vocab_size, txt_maxlen,
                                image_height, image_width, image_channels, max_video_frame_number,
                                video_height, video_width, video_channels, label_number)
