import tensorflow as tf
import numpy as np
import cv2
import random
import pickle
import os
import pathlib

from tensorflow import keras
import tensorflow_datasets as tfds
from dataset_public_interface import multimodal_data_path_generator, video_label_to_id


def store_list_to_pickle_file(data_list, store_file_name="data_list.pickle"):
    with open(store_file_name, 'wb') as wf:
        pickle.dump(data_list, wf)


def read_list_from_pikle_file(store_file_name="data_list.pickle"):
    with open(store_file_name, 'rb') as rf:
        data_list = pickle.load(rf)
        return data_list


def get_text_list_from_raw_txt_file(data_root="MP4_download"):
    data_root = pathlib.Path(data_root)
    all_txt_data_paths = [str(path) for path in
                          list(data_root.glob('*/*/*.txt'))]  # [MP4_download/360VR/89422838/89422838.txt,...]
    text_list = []
    for text_data_path in all_txt_data_paths:
        description_information_dict = eval(open(text_data_path).read())
        txt_brief = description_information_dict['mp4_txt_brief']
        text_list.append(txt_brief)
    return text_list


def tfds_text_encoder(text_list, filename_prefix='text_encoder', check_encoder=True):
    """
    TensorFlow dataset encoder
    """
    tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()

    for text in text_list:
        some_tokens = tokenizer.tokenize(text)
        vocabulary_set.update(some_tokens)

    vocab_size = len(vocabulary_set)
    print("vocab_size", vocab_size)

    text_encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    if check_encoder:
        example_text = 'I am the blogger of Wangjiang Artificial Think Tank.' \
                       ' Welcome to https://yuanxiaosc.github.io./'
        encoded_example = text_encoder.encode(example_text)
        decoded_example = text_encoder.decode(encoded_example)
        print("example_text:\t", example_text)
        print("encoded_example:\t", encoded_example)
        print("decoded_example:\t", decoded_example)

    text_encoder.save_to_file(filename_prefix)
    return text_encoder


def get_text_encoder(data_root='MP4_download', txt_encoder_filename_prefix='text_encoder'):
    if not os.path.exists(txt_encoder_filename_prefix + ".tokens"):
        print("Create text_encoder from raw text")
        text_list = get_text_list_from_raw_txt_file(data_root)
        text_encoder = tfds_text_encoder(text_list)
    else:
        print("TokenTextEncoder.load_from_file(txt_encoder_filename_prefix)")
        text_encoder = tfds.features.text.TokenTextEncoder.load_from_file(txt_encoder_filename_prefix)
    return text_encoder


def get_multimodal_data_path_and_label_list(data_root="MP4_download",
                                            store_file_name="store_file_name",
                                            shuffle_data=False):
    """
    Get the data path and label list
    :param data_root: Data storage location
    :param store_file_name: Data caching file
    :param shuffle_data: Disturbing the order of data
    :return: video_path_list, image_path_list, text_path_list, label_id_list
    """

    def load_raw_data_list(data_root, store_file_name):
        """
        Preferred use of cached data files
        """
        if os.path.exists(store_file_name) and os.path.exists(data_root):
            print("load data path list from pickle file!")
            multimodal_data_path_and_label_list = read_list_from_pikle_file(store_file_name)
        else:
            print(f"load data list from {data_root}")
            multimodal_data_path_and_label_list = list(multimodal_data_path_generator(data_root))
            store_list_to_pickle_file(multimodal_data_path_and_label_list, store_file_name)
        return multimodal_data_path_and_label_list

    multimodal_data_path_and_label_list = load_raw_data_list(data_root, store_file_name)

    if shuffle_data:
        random.shuffle(multimodal_data_path_and_label_list)

    video_path_list, image_path_list, text_path_list, label_id_list = [], [], [], []
    for video_path, image_path, text_path, label in multimodal_data_path_and_label_list:
        video_path_list.append(video_path)
        image_path_list.append(image_path)
        text_path_list.append(text_path)
        label_id_list.append(video_label_to_id[label])
    return video_path_list, image_path_list, text_path_list, label_id_list


def parser_multimodal_dataset(video, image, text, label):
    return {"video_data": video, "image_data": image, "text_data": text}, label


def multimodal_dataset_tf2(BATCH_SIZE=64, EPOCHS=100,
                           data_root='MP4_download', cache_file_name='cache_data_list.pickle',
                           txt_encoder_filename_prefix='text_encoder', txt_max_len=25,
                           max_video_frame_number=16, video_height=360, video_width=640,
                           image_height=270, image_width=480, shuffle_buffer_size=10):
    def parser_video_map_fn(video_path):
        def process_video_py_function(video_path):
            videoCapture = cv2.VideoCapture(video_path.numpy().decode())
            success, frame = videoCapture.read()
            frame_list = []
            frame_number = 0
            while success:
                if frame is None:
                    break
                if isinstance(max_video_frame_number, int) and frame_number == max_video_frame_number:
                    break
                image_np = frame
                resize_image_np = cv2.resize(image_np, dsize=(video_width, video_height))
                resize_image_np_expanded = np.expand_dims(resize_image_np, axis=0)
                frame_list.append(resize_image_np_expanded)
                frame_number += 1
                success, frame = videoCapture.read()
            encode_video = np.concatenate(frame_list, axis=0)
            return encode_video

        return tf.py_function(process_video_py_function,
                              inp=[video_path],
                              Tout=(tf.float32))

    def parser_text_map_fn(text_path):
        def process_text_py_function(text_path):
            text_path = text_path.numpy().decode()
            description_information_dict = eval(open(text_path).read())
            encode_txt = text_encoder.encode(description_information_dict['mp4_txt_brief'])
            encode_txt = keras.preprocessing.sequence.pad_sequences(
                [encode_txt], maxlen=txt_max_len, dtype='int32', padding='post', truncating='post', value=0.0)
            encode_txt = tf.cast(encode_txt[0], dtype=tf.int32)
            return encode_txt

        return tf.py_function(process_text_py_function,
                              inp=[text_path],
                              Tout=(tf.int32))

    def parser_image_data_map_fn(jpeg_file_path):
        """
        Read the picture data and specify the value in the [-1,1] range
        """
        image = tf.io.read_file(jpeg_file_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, [image_height, image_width])
        image = tf.cast(image, dtype=tf.float32)
        image = (image / 127.5) - 1.0
        return image

    def filter_video_data(video_data, image_data, text_data, label):
        """
        Filtered video is not equal to the specified(max_video_frame_number) number of frames
        """
        video_frame_number = tf.shape(video_data)[0]
        return tf.math.equal(video_frame_number, max_video_frame_number)

    # load data path list
    video_path_list, image_path_list, text_path_list, label_id_list = get_multimodal_data_path_and_label_list(
        data_root, cache_file_name, shuffle_data=True)

    # load text_encoder
    text_encoder = get_text_encoder(data_root, txt_encoder_filename_prefix)

    video_dataset = tf.data.Dataset.from_tensor_slices(video_path_list)
    video_dataset = video_dataset.map(parser_video_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    image_dataset = tf.data.Dataset.from_tensor_slices(image_path_list)
    image_dataset = image_dataset.map(parser_image_data_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    text_dataset = tf.data.Dataset.from_tensor_slices(text_path_list)
    text_dataset = text_dataset.map(parser_text_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    label_dataset = tf.data.Dataset.from_tensor_slices(label_id_list)

    multimodal_dataset = tf.data.Dataset.zip((video_dataset, image_dataset, text_dataset, label_dataset))
    multimodal_dataset = multimodal_dataset.filter(filter_video_data)
    multimodal_dataset = multimodal_dataset.map(parser_multimodal_dataset,
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    multimodal_dataset = multimodal_dataset.batch(BATCH_SIZE)
    multimodal_dataset = multimodal_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    multimodal_dataset = multimodal_dataset.shuffle(buffer_size=shuffle_buffer_size)
    multimodal_dataset = multimodal_dataset.repeat(EPOCHS)

    return multimodal_dataset


def check_multimodal_dataset(multimodal_dataset):
    for feature_dict, label in multimodal_dataset:
        print(feature_dict['video_data'].shape, feature_dict['video_data'].dtype)
        print(feature_dict['image_data'].shape, feature_dict['image_data'].dtype)
        print(feature_dict['text_data'].shape, feature_dict['text_data'].dtype)
        print(label.shape, label.dtype)
        print("")

def check_multimodal_dataset_2(multimodal_dataset):
    i = 0
    for encode_video, image, encoded_text, encode_label in multimodal_dataset:
        print(f"{i}")
        print(encode_video.shape, encode_video.dtype)
        print(image.shape, image.dtype)
        print(encoded_text.shape, encoded_text.dtype)
        print(encode_label.shape, encode_label.dtype)
        i += 1

def check_a_dataset(dataset, take_samples_number=5):
    for item_data in dataset.take(take_samples_number):
        print(item_data.shape, item_data.dtype)
        print("")


if __name__ == "__main__":
    BATCH_SIZE = 10
    EPOCHS = 100

    data_root = "/home/b418a/disk1/jupyter_workspace/yuanxiao/douyin/xinpianchang/MP4_download"
    fake_data_root = "/home/b418a/disk1/pycharm_room/yuanxiao/my_lenovo_P50s/Multimodal-short-video-dataset-and-baseline-model/MP4_download"
    store_file_name = "data_list.pickle"

    txt_encoder_filename_prefix = 'text_encoder'
    txt_max_len = 25

    max_video_frame_number = 16
    video_width = 640
    video_height = 360

    image_height = 270
    image_width = 480

    shuffle_buffer_size = 100

    multimodal_dataset = multimodal_dataset_tf2(BATCH_SIZE, EPOCHS,
                                                data_root, store_file_name,
                                                txt_encoder_filename_prefix, txt_max_len,
                                                max_video_frame_number, video_height, video_width,
                                                image_height, image_width,
                                                shuffle_buffer_size)

    i = 0
    for encode_video, image, encoded_text, encode_label in multimodal_dataset:
        print(f"{i}")
        print(encode_video.shape, encode_video.dtype)
        print(image.shape, image.dtype)
        print(encoded_text.shape, encoded_text.dtype)
        print(encode_label.shape, encode_label.dtype)
        i += 1
