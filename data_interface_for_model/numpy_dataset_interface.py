from tensorflow_dataset_interface import multimodel_numpy_data_interface

if __name__=="__main__":
    data_root = "/home/b418a/disk1/jupyter_workspace/yuanxiao/douyin/xinpianchang/MP4_download"
    fake_data_root = "/home/b418a/disk1/pycharm_room/yuanxiao/my_lenovo_P50s/Multimodal-short-video-dataset-and-baseline-model/MP4_download"

    shuffle_data = True
    BATCH_SIZE = 5
    REPEAT_DATASET = None

    txt_maxlen = 20
    image_height = 270
    image_width = 480
    max_video_frame_number = 100
    video_height = 360
    video_width = 640

    numpy_generator = multimodel_numpy_data_interface(fake_data_root, shuffle_data, BATCH_SIZE, REPEAT_DATASET,
                                                       txt_maxlen, image_height, image_width,
                                                       max_video_frame_number, video_height, video_width)

    for encode_video, encode_image, encoded_text, encode_label in numpy_generator:
        print("")
        print("encode_video", encode_video.shape, encode_video.dtype)
        print("encode_image", encode_image.shape, encode_image.dtype)
        print("encoded_text", encoded_text.shape, encoded_text.dtype)
        print("encode_label", encode_label.shape, encode_label.dtype)