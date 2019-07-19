import os
import random
import pathlib

import cv2
import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds

video_type_dict = {'360VR': 'VR', '4k': '4K', 'Technology': '科技', 'Sport': '运动', 'Timelapse': '延时',
                   'Aerial': '航拍', 'Animals': '动物', 'Sea': '大海', 'Beach': '海滩', 'space': '太空',
                   'stars': '星空', 'City': '城市', 'Business': '商业', 'Underwater': '水下摄影',
                   'Wedding': '婚礼', 'Archival': '档案', 'Backgrounds': '背景', 'Alpha Channel': '透明通道',
                   'Intro': '开场', 'Celebration': '庆典', 'Clouds': '云彩', 'Corporate': '企业',
                   'Explosion': '爆炸', 'Film': '电影镜头', 'Green Screen': '绿幕', 'Military': '军事',
                   'Nature': '自然', 'News': '新闻', 'R3d': 'R3d', 'Romantic': '浪漫', 'Abstract': '抽象'}

video_type_list = ['360VR', '4k', 'Abstract', 'Aerial', 'Alpha Channel', 'Animals', 'Archival', 'Backgrounds', 'Beach',
                   'Business', 'Celebration', 'City', 'Clouds', 'Corporate', 'Explosion', 'Film', 'Green Screen',
                   'Intro', 'Military', 'Nature', 'News', 'R3d', 'Romantic', 'Sea', 'Sport', 'Technology', 'Timelapse',
                   'Underwater', 'Wedding', 'space', 'stars']

video_label_to_id = {'360VR': 0, '4k': 1, 'Abstract': 2, 'Aerial': 3, 'Alpha Channel': 4, 'Animals': 5, 'Archival': 6,
                     'Backgrounds': 7, 'Beach': 8, 'Business': 9, 'Celebration': 10, 'City': 11, 'Clouds': 12,
                     'Corporate': 13, 'Explosion': 14, 'Film': 15, 'Green Screen': 16, 'Intro': 17, 'Military': 18,
                     'Nature': 19, 'News': 20, 'R3d': 21, 'Romantic': 22, 'Sea': 23, 'Sport': 24, 'Technology': 25,
                     'Timelapse': 26, 'Underwater': 27, 'Wedding': 28, 'space': 29, 'stars': 30}


def standardization_of_file_names(data_root="MP4_download"):
    """
    Uniform naming format for each set of data as follows:

    multimodal_data_id
        multimodal_data_id.jepg
        multimodal_data_id.mp4
        multimodal_data_id.txt
    """

    # Get all multimodal data type names
    data_root = pathlib.Path(data_root)
    label_names_list = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print(f"data_root contain video type numbers {len(label_names_list)}")
    print(f"data_root contain video type {label_names_list}")

    # Processing multimodal data sequentially
    for label_name in label_names_list:
        # Get all folders under a certain type of multimodal data
        label_mode = label_name + "/*"
        multimodal_data_dir = list(data_root.glob(label_mode))
        multimodal_data_dir = [str(path) for path in multimodal_data_dir]

        # File name for standardized multimodal data
        for multimodal_data_path in multimodal_data_dir:
            multimodal_data_id = os.path.basename(multimodal_data_path)
            for item_file in os.listdir(multimodal_data_path):
                item_file = os.path.join(multimodal_data_path, item_file)
                if item_file.endswith('.txt'):
                    os.rename(item_file, os.path.join(multimodal_data_path, multimodal_data_id + ".txt"))
                elif item_file.endswith('.jpeg'):
                    os.rename(item_file, os.path.join(multimodal_data_path, multimodal_data_id + ".jpeg"))
                elif item_file.endswith('.mp4'):
                    os.rename(item_file, os.path.join(multimodal_data_path, multimodal_data_id + ".mp4"))
                elif item_file.endswith('.ipynb_checkpoints'):
                    pass
                else:
                    raise ValueError("An abnormal document appeared! check!")


def get_filtered_all_multimodal_data_item_file_dir_list(data_root="MP4_download"):
    """
    :param data_root: Original file root path
    :return: filtered_all_multimodal_data_item_file_dir_list
         ['data_root/360VR/89422838', 'data_root/360VR/107178375', 'data_root/360VR/67370207']
    """

    def delete_incomplete_data(multimodal_data_item_file_dir):
        multimodal_data_id = os.path.basename(multimodal_data_item_file_dir)

        txt_file_path = os.path.join(multimodal_data_item_file_dir, multimodal_data_id + ".txt")
        jpeg_file_path = os.path.join(multimodal_data_item_file_dir, multimodal_data_id + ".jpeg")
        mp4_file_path = os.path.join(multimodal_data_item_file_dir, multimodal_data_id + ".mp4")

        for file_path in [mp4_file_path, jpeg_file_path, txt_file_path]:
            if not os.path.exists(file_path):
                return False
        return True

    # Get all multimodal data type names
    data_root = pathlib.Path(data_root)
    label_names_list = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

    all_multimodal_data_item_file_dir_list = list()
    for label_name in label_names_list:
        # Get all folders under a certain type of multimodal data
        label_mode = label_name + "/*"
        multimodal_data_dir = list(data_root.glob(label_mode))
        multimodal_data_dir = [str(path) for path in multimodal_data_dir]

        all_multimodal_data_item_file_dir_list.extend(multimodal_data_dir)
    print("all_multimodal_data_item_file_dir_list length", len(all_multimodal_data_item_file_dir_list))

    filtered_all_multimodal_data_item_file_dir_list = list(
        filter(delete_incomplete_data, all_multimodal_data_item_file_dir_list))

    print("filtered_all_multimodal_data_item_file_dir_list length",
          len(filtered_all_multimodal_data_item_file_dir_list))

    return filtered_all_multimodal_data_item_file_dir_list


def get_description_information(txt_path):
    """description_information include: {'mp4_id': '', 'mp4_download_url': '', 'mp4_time': '',
    'mp4_background_image_url': '', 'mp4_txt_brief': ''}"""
    description_information_dict = eval(open(txt_path).read())
    return description_information_dict


def get_text_list_from_raw_txt_file(data_root="MP4_download"):
    """
    Getting mp4_txt_brief text data from the original file
    :param data_root:  Original file root path
    :return: text_list
    """
    data_root = pathlib.Path(data_root)
    all_txt_data_paths = [str(path) for path in
                          list(data_root.glob('*/*/*.txt'))]  # [MP4_download/360VR/89422838/89422838.txt,...]
    text_list = []
    for text_data_path in all_txt_data_paths:
        description_information_dict = eval(open(text_data_path).read())
        txt_brief = description_information_dict['mp4_txt_brief']
        text_list.append(txt_brief)
    return text_list


def tfds_text_encoder_and_word_set(text_list):
    """
    TensorFlow dataset encoder
    :param text_list:
    :return:
    """
    tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()

    for text in text_list:
        some_tokens = tokenizer.tokenize(text)
        vocabulary_set.update(some_tokens)

    vocab_size = len(vocabulary_set)
    print("vocab_size", vocab_size)

    text_encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    example_text = 'I am the blogger of Wangjiang Artificial Think Tank.' \
                   ' Welcome to https://yuanxiaosc.github.io./'
    encoded_example = text_encoder.encode(example_text)
    print("example_text:\t", example_text)
    print("encoded_example:\t", encoded_example)

    return text_encoder, vocabulary_set


def multimodal_data_path_generator(data_root="MP4_download", shuffle_data=False):
    """
    Multimodal Data Path Generator
    :param data_root:  Original file root path
    :param shuffle_data: Disrupt data order
    :return:data_path_generator

    Usage method:
    for mp4_file_path, jpeg_file_path, txt_file_path, label in multimodal_data_path_generator(data_root,
                                                                                              shuffle_data):
        print("")
        print("mp4_file_path", mp4_file_path)
        print("jpeg_file_path", jpeg_file_path)
        print("txt_file_path", txt_file_path)
        print("label", label)
    """
    multimodal_data_item_file_dir_list = get_filtered_all_multimodal_data_item_file_dir_list(data_root)

    if shuffle_data:
        random.shuffle(multimodal_data_item_file_dir_list)

    for item_file_dir in multimodal_data_item_file_dir_list:  # data_root/Business/849
        multimodal_data_id = os.path.basename(item_file_dir)  # 849
        label = os.path.basename(os.path.dirname(item_file_dir))  # Business
        txt_file_path = os.path.join(item_file_dir, multimodal_data_id + ".txt")  # data_root/Business/849/849.txt
        jpeg_file_path = os.path.join(item_file_dir, multimodal_data_id + ".jpeg")  # data_root/Business/849/849.jpeg
        mp4_file_path = os.path.join(item_file_dir, multimodal_data_id + ".mp4")  # data_root/Business/849/849.mp4

        # yield data_root/Business/849/849.mp4, data_root/Business/849/849.jpeg, data_root/Business/849/849.txt, Business
        yield mp4_file_path, jpeg_file_path, txt_file_path, label


def get_multimodal_data_path_list(data_root="MP4_download", shuffle_data=False):
    """
    Getting a multimodal data path list
    :param data_root:  Original file root path
    :param shuffle_data: Disrupt data order
    :return:
    """
    multimodal_data_path_list = [(mp4_file_path, jpeg_file_path, txt_file_path) for
                                 mp4_file_path, jpeg_file_path, txt_file_path, label in
                                 multimodal_data_path_generator(data_root, shuffle_data)]
    return multimodal_data_path_list


def multimodal_encode_data_generator(data_root="MP4_download", shuffle_data=False, txt_maxlen=25,
                                     max_video_frame_number=None, video_width=640, video_height=360):
    """
    Multimodal Encode Data Generator
    :param data_root:  Original file root path
    :param shuffle_data: Disrupt data order
    :param max_video_frame_number: None -> keep all video number, int -> max need video frame number
    :return: multimodal_encode_data_generator

    Usage method:
    for encode_video, image_file_path, encode_txt, encode_label in encode_multimodal_data(fake_data_root,
                                                                                          shuffle_data,
                                                                                          max_video_frame_number):
        print("")
        print("encode_video.shape", encode_video.shape)
        print("image_file_path", image_file_path)
        print("encode_txt", encode_txt)
        print("encode_label", encode_label)
    """

    text_list = get_text_list_from_raw_txt_file(data_root)
    text_encoder, vocabulary_set = tfds_text_encoder_and_word_set(text_list)

    def process_video(video_file_path, max_video_frame_number=None, video_width=640, video_height=360):
        videoCapture = cv2.VideoCapture(video_file_path)
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

    def process_image_data(label):
        encode_label = video_label_to_id[label]
        return encode_label

    def process_txt_data(txt_file_path, txt_maxlen=25):
        description_information_dict = get_description_information(txt_file_path)
        encode_txt = text_encoder.encode(description_information_dict['mp4_txt_brief'])
        encode_txt = keras.preprocessing.sequence.pad_sequences(
            [encode_txt], maxlen=txt_maxlen, dtype='int32', padding='post', truncating='post', value=0.0)
        return encode_txt[0]

    for mp4_file_path, jpeg_file_path, txt_file_path, label in multimodal_data_path_generator(data_root, shuffle_data):
        encode_video = process_video(mp4_file_path, max_video_frame_number, video_width, video_height)
        image_file_path = jpeg_file_path
        encode_label = process_image_data(label)
        encode_txt = process_txt_data(txt_file_path, txt_maxlen)
        yield encode_video, image_file_path, encode_txt, encode_label


if __name__ == "__main__":
    data_root = "/home/b418a/disk1/jupyter_workspace/yuanxiao/douyin/xinpianchang/MP4_download"
    fake_data_root = "/home/b418a/disk1/pycharm_room/yuanxiao/my_lenovo_P50s/Multimodal-short-video-dataset-and-baseline-model/MP4_download"
    standardized_file_name = False  # Only need to be executed once, format the path of the original download file
    shuffle_data = True

    txt_maxlen = 25
    max_video_frame_number = 100
    video_height = 360
    video_width = 640

    if standardized_file_name:
        standardization_of_file_names(data_root)

    for mp4_file_path, jpeg_file_path, txt_file_path, label in multimodal_data_path_generator(fake_data_root,
                                                                                              shuffle_data):
        print("")
        print("mp4_file_path", mp4_file_path)
        print("jpeg_file_path", jpeg_file_path)
        print("txt_file_path", txt_file_path)
        print("label", label)

    for encode_video, image_file_path, encode_txt, encode_label in multimodal_encode_data_generator(fake_data_root,
                                                                                                    shuffle_data,
                                                                                                    txt_maxlen,
                                                                                                    max_video_frame_number,
                                                                                                    video_width,
                                                                                                    video_height):
        print("")
        print("encode_video.shape", encode_video.shape)
        print("image_file_path", image_file_path)
        print("encode_txt.shape", encode_txt.shape)
        print("encode_txt", encode_txt)
        print("encode_label", encode_label)
