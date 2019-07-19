import os
import sys
import pathlib
import pandas as pd
import json


def clean_specified_type_file(data_root=None, specified_type_list=["*/*/*.mp4", "*/*/*.jpeg", "*/*/*.txt"]):
    """
    :param data_root: To delete the root of the file
    :param specified_type_list: To delete the relationship between the specified file and the root directory
    """
    if data_root is None:
        data_root = os.getcwd()

    data_root = pathlib.Path(data_root)
    garbage_file_list = list()
    # get the paths to clear the file
    for t in specified_type_list:
        names_list = sorted(item.name for item in data_root.glob(t))
        garbage_file_list.extend(names_list)

    # remove file
    for name in garbage_file_list:
        os.remove(name)


def get_description_information(txt_path):
    """description_information include: {'mp4_id': '', 'mp4_download_url': '', 'mp4_time': '',
    'mp4_background_image_url': '', 'mp4_txt_brief': ''}"""
    description_information_dict = eval(open(txt_path).read())
    return description_information_dict


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


def count_file_number(data_root="MP4_download"):
    """
    statistics files number
    :return {'Military': 18560, 'Business': 19200, 'Archival': 10176, 'Romantic': 19162,...}
             all number:      56xx42
    """
    video_label_number = len(os.listdir(data_root))
    print("video_label_number:\t", video_label_number)
    multimodal_data_number_dict = dict()
    all_number = 0
    for video_label in os.listdir(data_root):
        video_label_dir = os.path.join(data_root, video_label)
        # print("video_label_dir:\t", video_label_dir)
        multimodal_data_number = len(os.listdir(video_label_dir))
        # print("multimodal_data_number:\t", multimodal_data_number)
        multimodal_data_number_dict[video_label] = multimodal_data_number
        all_number += multimodal_data_number
    print(multimodal_data_number_dict)
    print("all number:\t", all_number)
    return multimodal_data_number_dict


def statistics_all_multimodal_data_information_to_json_file(data_root="MP4_download",
                                                            store_multimodal_info_json_file_path="multimodal_data_info.json"):
    """
    data_root all *.txt files to a *.json file
    """
    data_root = pathlib.Path(data_root)
    all_txt_data_paths = [str(path) for path in
                          list(data_root.glob('*/*/*.txt'))]  # [MP4_download/360VR/89422838/89422838.txt,...]

    json_write_f = open(store_multimodal_info_json_file_path, "w", encoding='utf-8')
    for text_data_path in all_txt_data_paths:
        video_label_path = os.path.dirname(os.path.dirname(text_data_path))  # /MP4_download/360VR/
        video_label = os.path.basename(video_label_path)  # 360VR

        description_information_dict = get_description_information(text_data_path)
        description_information_dict["video_label"] = video_label

        line_json = json.dumps(description_information_dict, ensure_ascii=False)
        json_write_f.write(line_json + "\n")
    json_write_f.close()


def read_multimodal_data_information_json_file(json_file_path="multimodal_data_info.json"):
    """
    :param json_file_path:
    :return: multimodal_data_information_list
            [{'mp4_id': '97930081', 'mp4_download_url': ...'video_label': 'Military'},
            {'mp4_id': '64413672', 'mp4_download_url': ... 'video_label': 'Military'}]
    """

    def check_data(line_dict):
        for item in ['mp4_id', 'video_label', 'mp4_time', 'mp4_download_url', 'mp4_background_image_url',
                     'mp4_txt_brief']:
            if item not in line_dict:
                return False
        return True

    multimodal_data_information_list = list()
    with open(json_file_path, 'r', encoding='utf-8') as f:
        try:
            while True:
                line = f.readline()
                if line:
                    line_dict = json.loads(line)
                    if check_data(line_dict):
                        multimodal_data_information_list.append(line_dict)
                    else:
                        print("incomplete data:")
                        print(line_dict)
                else:
                    break
        except:
            f.close()
    return multimodal_data_information_list


def multimodal_data_json_file_to_datafram(json_file_path="multimodal_data_info.json"):
    """
    json file to pandas.DataFrame
    """
    if not os.path.exists(json_file_path):
        print("python statistics_all_multimodal_data_information_to_json_file(data_root, json_file_path)")
        raise ValueError("Not found json file!")

    multimodal_data_information_list = read_multimodal_data_information_json_file(json_file_path)

    multimodal_data_information_dict = {'mp4_id': [], 'video_label': [], 'mp4_time': [],
                                        'mp4_download_url': [], 'mp4_background_image_url': [], 'mp4_txt_brief': []}

    for data in multimodal_data_information_list:
        multimodal_data_information_dict['mp4_id'].append(data['mp4_id'])
        multimodal_data_information_dict['video_label'].append(data['video_label'])
        multimodal_data_information_dict['mp4_time'].append(data['mp4_time'])
        multimodal_data_information_dict['mp4_download_url'].append(data['mp4_download_url'])
        multimodal_data_information_dict['mp4_background_image_url'].append(data['mp4_background_image_url'])
        multimodal_data_information_dict['mp4_txt_brief'].append(data['mp4_txt_brief'])

    multimodal_data_information_datafram = pd.DataFrame(multimodal_data_information_dict)

    return multimodal_data_information_datafram


def aggravate_data_utils_main(data_root, json_file_path="./multimodal_data_info.json"):
    """
    aggregate_download_data_to_a_json_file
    :param data_root: download files data root
    :param json_file_path: produce json file path
    :return:
    """
    # standard file name
    standardization_of_file_names(data_root)

    # produce json file
    statistics_all_multimodal_data_information_to_json_file(data_root, json_file_path)

    # analysis json file
    multimodal_data_information_datafram = multimodal_data_json_file_to_datafram(json_file_path)
    print(multimodal_data_information_datafram.describe())


if __name__ == "__main__":
    data_root = "/home/b418a/disk1/jupyter_workspace/yuanxiao/douyin/xinpianchang/MP4_download"
    json_file_path = "./multimodal_data_info.json"

    if len(sys.argv) == 3:
        data_root = sys.argv[1]
        json_file_path = sys.argv[2]
    elif len(sys.argv) == 2:
        data_root = sys.argv[1]

    aggravate_data_utils_main(data_root, json_file_path)
