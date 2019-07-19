import pathlib
import os
import random
import matplotlib.pyplot as plt


def get_video_type(dir_name="MP4_download"):
    """
    :param dir_name:
    :return: video_type: , example: {
                   '360VR': 'VR', '4k': '4K', 'Technology': '科技', 'Sport': '运动', 'Timelapse': '延时',
                   'Aerial': '航拍', 'Animals': '动物', 'Sea': '大海', 'Beach': '海滩', 'space': '太空',
                   'stars': '星空', 'City': '城市', 'Business': '商业', 'Underwater': '水下摄影',
                   'Wedding': '婚礼', 'Archival': '档案', 'Backgrounds': '背景', 'Alpha Channel': '透明通道',
                   'Intro': '开场', 'Celebration': '庆典', 'Clouds': '云彩', 'Corporate': '企业',
                   'Explosion': '爆炸', 'Film': '电影镜头', 'Green Screen': '绿幕', 'Military': '军事',
                   'Nature': '自然', 'News': '新闻', 'R3d': 'R3d', 'Romantic': '浪漫', 'Abstract': '抽象'}
    """
    dir_name = pathlib.Path(dir_name)
    video_type = list(dir_name.glob('*'))
    video_type = [str(path).split("/")[-1] for path in video_type]
    print("Existing Video Types Numbers:\t", len(video_type))
    print("Existing Video Types        :\t", video_type)
    print("")
    return video_type


def get_description_information(txt_path):
    """description_information include: {'mp4_id': '', 'mp4_download_url': '', 'mp4_time': '',
    'mp4_background_image_url': '', 'mp4_txt_brief': ''}"""
    description_information_dict = eval(open(txt_path).read())
    return description_information_dict


def show_image_and_description_information(image_path, description_information_dict):
    lena = plt.imread(image_path)
    plt.imshow(lena)
    plt.title(description_information_dict["mp4_background_image_url"])
    plt.xlabel(description_information_dict["mp4_txt_brief"])
    plt.ylabel(description_information_dict["mp4_id"])
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.show()


def check_download_file(dir_name="MP4_download", video_type=None, shuffle_data=False,
                        print_file_path=False, show_txt=False, show_image=False, check_number=None, ):
    """
    Check one type of video file downloaded from https://www.xinpianchang.com/
    :param dir_name: Root directory for storing data
    :param video_type: Check video type, example: {
                   '360VR': 'VR', '4k': '4K', 'Technology': '科技', 'Sport': '运动', 'Timelapse': '延时',
                   'Aerial': '航拍', 'Animals': '动物', 'Sea': '大海', 'Beach': '海滩', 'space': '太空',
                   'stars': '星空', 'City': '城市', 'Business': '商业', 'Underwater': '水下摄影',
                   'Wedding': '婚礼', 'Archival': '档案', 'Backgrounds': '背景', 'Alpha Channel': '透明通道',
                   'Intro': '开场', 'Celebration': '庆典', 'Clouds': '云彩', 'Corporate': '企业',
                   'Explosion': '爆炸', 'Film': '电影镜头', 'Green Screen': '绿幕', 'Military': '军事',
                   'Nature': '自然', 'News': '新闻', 'R3d': 'R3d', 'Romantic': '浪漫', 'Abstract': '抽象'}
    :param shuffle_data: Scrambling data, sampling check
    :param print_file_path: Print out all file paths
    :param show_txt: Print video meta information
    :param show_image: Show video cover image
    :param check_number: Number of files to check, None stands for all
    :return: Number of various documents (all_item_number, txt_number, image_number, video_number)
    """
    dir_name = pathlib.Path(dir_name)

    path_mode = video_type + "/*"
    all_item_paths = list(dir_name.glob(path_mode))
    all_item_paths = [str(path) for path in all_item_paths]
    if shuffle_data:
        random.shuffle(all_item_paths)
    all_item_number = len(all_item_paths)
    txt_number = 0
    image_number = 0
    video_number = 0

    for idx, item in enumerate(all_item_paths):
        item_id = item.split("/")[-1]
        item_type = item.split("/")[1]
        for item_file in os.listdir(item):
            if item_file.endswith('.txt'):
                txt_path = os.path.join(item, item_file)
            elif item_file.endswith('.jpeg'):
                image_path = os.path.join(item, item_file)
            elif item_file.endswith('.mp4'):
                mp4_path = os.path.join(item, item_file)
            else:
                raise ValueError("An abnormal document appeared! check!")

        if os.path.exists(txt_path):
            description_information_dict = get_description_information(txt_path)
        else:
            description_information_dict = {'mp4_id': '', 'mp4_download_url': '', 'mp4_time': '',
                                            'mp4_background_image_url': '', 'mp4_txt_brief': ''}

        if os.path.exists(txt_path):
            if print_file_path:
                print(f"exsit {txt_path}")
            txt_number += 1
            if show_txt:
                print(open(txt_path).read())
                print("item_type:\t", item_type)
        else:
            if print_file_path:
                print(f"Not exsit {txt_path}")

        if os.path.exists(image_path):
            if print_file_path:
                print(f"exsit {image_path}")
            image_number += 1
            if show_image:
                show_image_and_description_information(image_path, description_information_dict)
        else:
            if print_file_path:
                print(f"Not exsit {image_path}")

        if os.path.exists(mp4_path):
            if print_file_path:
                print(f"exists {mp4_path}")
            video_number += 1
        else:
            if print_file_path:
                print(f"Not exists {mp4_path}")

        if print_file_path:
            print("")

        if check_number is not None:
            if idx == check_number - 1:
                break

    count_item_number_list = [all_item_number, txt_number, image_number, video_number]
    if len(set(count_item_number_list)) == 1:
        print("All documents are complete!")
    else:
        print("Document missing!")
    print("all_item_number:\t", all_item_number)
    print("txt_number:\t", txt_number)
    print("image_number:\t", image_number)
    print("video_number:\t", video_number)
    return all_item_number, txt_number, image_number, video_number


def check_all_downloaded_files(dir_name="MP4_download"):
    """Check all files downloaded from https://www.xinpianchang.com/"""
    for mp4_type in get_video_type(dir_name=dir_name):
        print(f"video_type\t:{mp4_type}")
        check_download_file(dir_name=dir_name, video_type=mp4_type, shuffle_data=False, print_file_path=False, show_txt=False, show_image=False, check_number=None)
        print(" ")