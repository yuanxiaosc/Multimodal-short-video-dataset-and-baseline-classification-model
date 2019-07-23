import pathlib
import os
import tensorflow_datasets as tfds


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


if __name__=="__main__":
    data_root = "/home/b418a/disk1/jupyter_workspace/yuanxiao/douyin/xinpianchang/MP4_download"
    fake_data_root = "/home/b418a/disk1/pycharm_room/yuanxiao/my_lenovo_P50s/Multimodal-short-video-dataset-and-baseline-model/MP4_download"

    txt_encoder_filename_prefix = 'text_encoder'

    if not os.path.exists(txt_encoder_filename_prefix + ".tokens"):
        print("Create text_encoder from raw text")
        text_list = get_text_list_from_raw_txt_file(data_root)
        text_encoder = tfds_text_encoder(text_list)
    else:
        print("TokenTextEncoder.load_from_file(txt_encoder_filename_prefix)")
        text_encoder = tfds.features.text.TokenTextEncoder.load_from_file(txt_encoder_filename_prefix)

    print("text_encoder.vocab_size", text_encoder.vocab_size)



