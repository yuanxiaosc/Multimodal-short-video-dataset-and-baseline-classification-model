import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "data_interface_for_model")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "baseline_model")))
from baseline_model.mutimodal_baseline_model import create_multimodal_baseline_model
from data_interface_for_model.tfv2_interface import multimodal_dataset_tf2, get_text_encoder

BATCH_SIZE = 10
EPOCHS = 100

data_root = "/home/b418a/disk1/jupyter_workspace/yuanxiao/douyin/xinpianchang/MP4_download"
store_file_name = "data_list.pickle"

txt_encoder_filename_prefix = 'text_encoder'
# load text_encoder
text_encoder = get_text_encoder(data_root, txt_encoder_filename_prefix)

vocab_size = text_encoder.vocab_size + 1  # 1 for unknown
txt_max_len = 25

max_video_frame_number = 3
video_width = 640
video_height = 360
video_channels = 3

image_height = 270
image_width = 480
image_channels = 3

label_number = 31

LEARN_RATE = 0.001

shuffle_buffer_size = 20

checkpoint_path = "checkpoint"
log_path = "logs"
multimodal_dataset = multimodal_dataset_tf2(BATCH_SIZE, EPOCHS,
                                            data_root, store_file_name,
                                            txt_encoder_filename_prefix, txt_max_len,
                                            max_video_frame_number, video_height, video_width,
                                            image_height, image_width,
                                            shuffle_buffer_size)

multimodal_model = create_multimodal_baseline_model(label_number=label_number, txt_max_len=txt_max_len,
                                                    text_vocab_size=vocab_size, text_embedding_dim=100,
                                                    text_lstm_units=64, text_output_dim=50,
                                                    image_height=image_height, image_width=image_width,
                                                    image_channels=image_channels, image_output_dim=50,
                                                    max_video_frame_number=max_video_frame_number,
                                                    video_height=video_height, video_width=video_width,
                                                    video_channels=video_channels, video_output_dim=50)

multimodal_model.compile(optimizer=tf.keras.optimizers.Adam(LEARN_RATE),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Create callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,
                                                         verbose=1, save_freq='epoch')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
callback_list = [checkpoint_callback, tensorboard_callback]


ckpt = tf.train.Checkpoint(multimodal_model=multimodal_model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")
multimodal_model.fit(multimodal_dataset, epochs=EPOCHS, callbacks=callback_list)


