import os
import shutil

video_type_dict = {'360VR': 'VR', '4k': '4K', 'Technology': '科技', 'Sport': '运动', 'Timelapse': '延时',
                   'Aerial': '航拍', 'Animals': '动物', 'Sea': '大海', 'Beach': '海滩', 'space': '太空',
                   'stars': '星空', 'City': '城市', 'Business': '商业', 'Underwater': '水下摄影',
                   'Wedding': '婚礼', 'Archival': '档案', 'Backgrounds': '背景', 'Alpha Channel': '透明通道',
                   'Intro': '开场', 'Celebration': '庆典', 'Clouds': '云彩', 'Corporate': '企业',
                   'Explosion': '爆炸', 'Film': '电影镜头', 'Green Screen': '绿幕', 'Military': '军事',
                   'Nature': '自然', 'News': '新闻', 'R3d': 'R3d', 'Romantic': '浪漫', 'Abstract': '抽象'}


def make_fake_data(true_data_root, fake_data_root="./MP4_download", fake_video_number=1):
    """
    In order not to damage the original data, copy the original data for research
    """
    if not os.path.exists(fake_data_root):
        os.mkdir(fake_data_root)

    video_type_list = list(video_type_dict.keys())

    for multimodal_data_type in video_type_list[:fake_video_number]:
        true_multimodal_a_type_data_dir = os.path.join(true_data_root, multimodal_data_type)
        fake_multimodal_a_type_data_dir = os.path.join(fake_data_root, multimodal_data_type)
        shutil.copytree(true_multimodal_a_type_data_dir, fake_multimodal_a_type_data_dir)


if __name__=="__main__":
    true_data_root = "/home/b418a/disk1/jupyter_workspace/yuanxiao/douyin/xinpianchang/MP4_download"
    fake_data_root = "/home/b418a/disk1/pycharm_room/yuanxiao/my_lenovo_P50s/Multimodal-short-video-dataset-and-baseline-model/MP4_download"
    fake_video_number = 1
    make_fake_data(true_data_root, fake_data_root, fake_video_number)
