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

checkpoint_path = "manual_checkpoint"
log_path = "manual_logs"
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
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

ckpt = tf.train.Checkpoint(multimodal_model=multimodal_model,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'Latest checkpoint restored! start_epoch is {start_epoch}')

@tf.function
def train_step(x_batch_train, y_batch_train):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables autodifferentiation.
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        logits = multimodal_model([x_batch_train['text_data'],
                                   x_batch_train['image_data'],
                                   x_batch_train['video_data']])  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        loss_value = loss_fn(y_batch_train, logits)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, multimodal_model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, multimodal_model.trainable_weights))
    return loss_value

# Iterate over epochs.
for epoch in range(start_epoch, EPOCHS):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(multimodal_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 200 batches.
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
            print('Seen so far: %s samples' % ((step + 1) * 64))

        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
