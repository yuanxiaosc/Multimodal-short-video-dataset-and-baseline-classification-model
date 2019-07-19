import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import pathlib
from dataset_public_interface import multimodal_encode_data_generator


def multimodal_tensorflow_dataset(data_root, shuffle_data=False, BATCH_SIZE=100, REPEAT_DATASET=None,
                                  txt_maxlen=20, image_height=270, image_width=480,
                                  max_video_frame_number=100, video_height=360, video_width=640):
    """
    multimodal tensorflow dataset
    :param data_root:  Original file root path
    Usage method:
    multimodal_dataset = multimodal_tensorflow_dataset(fake_data_root, shuffle_data, BATCH_SIZE, REPEAT_DATASET,
                                                       txt_maxlen, image_height, image_width,
                                                       max_video_frame_number, video_height, video_width)

    i = 0
    for encode_video, image, encoded_text, encode_label in multimodal_dataset:
        print(f"{i}")
        print(encode_video.shape, encode_video.dtype)
        print(image.shape, image.dtype)
        print(encoded_text.shape, encoded_text.dtype)
        print(encode_label.shape, encode_label.dtype)
        i += 1
    """

    def filter_video_data(encode_video, image_file_path, encoded_text, encode_label):
        """
        Filtered video is not equal to the specified(max_video_frame_number) number of frames
        """
        video_frame_number = tf.shape(encode_video)[0]
        return tf.math.equal(video_frame_number, max_video_frame_number)

    def parser_multimodal_data(encode_video, image_file_path, encoded_text, encode_label):
        def parser_image_data(jpeg_file_path):
            """
            Read the picture data and specify the value in the [-1,1] range
            """
            image = tf.io.read_file(jpeg_file_path)
            image = tf.image.decode_jpeg(image)
            image = tf.image.resize(image, [image_height, image_width])
            image = tf.cast(image, dtype=tf.float32)
            image = (image / 127.5) - 1.0
            return image

        image = parser_image_data(image_file_path)
        return encode_video, image, encoded_text, encode_label

    multimodal_dataset = tf.data.Dataset.from_generator(
        lambda: multimodal_encode_data_generator(data_root, shuffle_data, txt_maxlen,
                                                 max_video_frame_number, video_width, video_height),
        output_shapes=(tf.TensorShape([None, video_height, video_width, 3]),
                       tf.TensorShape(None), tf.TensorShape(txt_maxlen), tf.TensorShape(())),
        output_types=(tf.float32, tf.string, tf.int32, tf.int32))

    multimodal_dataset = multimodal_dataset.map(parser_multimodal_data,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    multimodal_dataset = multimodal_dataset.filter(filter_video_data)
    multimodal_dataset = multimodal_dataset.repeat(REPEAT_DATASET)
    multimodal_dataset = multimodal_dataset.batch(BATCH_SIZE)
    multimodal_dataset = multimodal_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return multimodal_dataset


def multimodel_numpy_data_interface(data_root, shuffle_data=False, BATCH_SIZE=100, REPEAT_DATASET=None,
                                    txt_maxlen=20, image_height=270, image_width=480,
                                    max_video_frame_number=100, video_height=360, video_width=640):
    multimodal_dataset = multimodal_tensorflow_dataset(data_root, shuffle_data, BATCH_SIZE, REPEAT_DATASET,
                                                       txt_maxlen, image_height, image_width,
                                                       max_video_frame_number, video_height, video_width)

    for encode_video, encode_image, encoded_text, encode_label in multimodal_dataset:
        yield encode_video.numpy(), encode_image.numpy(), encoded_text.numpy(), encode_label.numpy()


if __name__ == "__main__":
    data_root = "/home/b418a/disk1/jupyter_workspace/yuanxiao/douyin/xinpianchang/MP4_download"
    fake_data_root = "/home/b418a/disk1/pycharm_room/yuanxiao/my_lenovo_P50s/Multimodal-short-video-dataset-and-baseline-model/MP4_download"

    shuffle_data = True
    BATCH_SIZE = 16
    REPEAT_DATASET = None

    txt_maxlen = 20
    image_height = 270
    image_width = 480
    max_video_frame_number = 100
    video_height = 360
    video_width = 640

    multimodal_dataset = multimodal_tensorflow_dataset(fake_data_root, shuffle_data, BATCH_SIZE, REPEAT_DATASET,
                                                       txt_maxlen, image_height, image_width,
                                                       max_video_frame_number, video_height, video_width)

    i = 0
    for encode_video, image, encoded_text, encode_label in multimodal_dataset:
        print(f"{i}")
        print(encode_video.shape, encode_video.dtype)
        print(image.shape, image.dtype)
        print(encoded_text.shape, encoded_text.dtype)
        print(encode_label.shape, encode_label.dtype)
        i += 1
