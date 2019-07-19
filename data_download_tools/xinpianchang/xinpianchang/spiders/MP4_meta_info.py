# -*- coding: utf-8 -*-
import scrapy
import random
import os
from bs4 import BeautifulSoup

video_type_dict = {'360VR': 'VR', '4k': '4K', 'Technology': '科技', 'Sport': '运动', 'Timelapse': '延时',
                   'Aerial': '航拍', 'Animals': '动物', 'Sea': '大海', 'Beach': '海滩', 'space': '太空',
                   'stars': '星空', 'City': '城市', 'Business': '商业', 'Underwater': '水下摄影',
                   'Wedding': '婚礼', 'Archival': '档案', 'Backgrounds': '背景', 'Alpha Channel': '透明通道',
                   'Intro': '开场', 'Celebration': '庆典', 'Clouds': '云彩', 'Corporate': '企业',
                   'Explosion': '爆炸', 'Film': '电影镜头', 'Green Screen': '绿幕', 'Military': '军事',
                   'Nature': '自然', 'News': '新闻', 'R3d': 'R3d', 'Romantic': '浪漫', 'Abstract': '抽象'}

def get_page_start_end_by_mp4_type(mp4_type):
	# Check https://resource.xinpianchang.com/video/list for the latest information
	# The update time is 2019/07/19
    if mp4_type in ["360VR"]:
        return 1, 18
    elif mp4_type in ["Archival"]:
        return 1, 170
    elif mp4_type in ["R3d"]:
        return 1, 264
    else:
        return 1, 301

class Mp4Spider(scrapy.Spider):
    name = 'MP4_meta_info'
    start_urls = ['https://www.xinpianchang.com/']
    custom_settings = {
        'DOWNLOAD_DELAY': 3.5,
        'DOWNLOAD_TIMEOUT': 180,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'JOBDIR': "reamin/MP4_meta_info_001"
    }

    def start_requests(self):
        self.MP4_base_dir = "MP4_download"
        accessed_url_file = "reamin/accessed_url.txt"
        if not os.path.exists(self.MP4_base_dir):
            os.mkdir(self.MP4_base_dir)
        video_type_list = list(video_type_dict.keys())
        random.shuffle(video_type_list)
        for mp4_type in video_type_list:
            if mp4_type not in ["Explosion", ]:
                mp4_type_store_dir = os.path.join(self.MP4_base_dir, mp4_type)
                if not os.path.exists(mp4_type_store_dir):
                    os.makedirs(mp4_type_store_dir)
                page_number_start, page_number_end = get_page_start_end_by_mp4_type(mp4_type)
                for page_number in range(page_number_start, page_number_end):
                    mp4_list_page_url = f"https://resource.xinpianchang.com/video/list?cate={mp4_type}&page={page_number}"
                    yield scrapy.Request(url=mp4_list_page_url, callback=self.parse_video_meta_info,
                                         meta={'mp4_type_store_dir': mp4_type_store_dir},
                                         headers={'Referer': 'https://www.xinpianchang.com/'})

    def parse_video_meta_info(self, response):
        mp4_type_store_dir = response.meta["mp4_type_store_dir"]
        bs = BeautifulSoup(response.body, "html.parser")
        for index, item in enumerate(
                bs.find_all("li", {"class": {"single-video J_sigle_video", "single-video J_sigle_video detail-more"}})):
            mp4_id = item["id"]
            mp4_download_url = item['data-preview']
            mp4_time = item.find_all("div", class_="single-video-duration")[0].string
            mp4_background_image_url = item.find_all("div", class_="thumb-img")[0]["style"][len("background-image:url("):-1]
            mp4_txt_brief = item.find_all("p", class_="single-brief J_single_brief")[0].string
            mp4_meta_info_dict = {"mp4_id": mp4_id, "mp4_download_url": mp4_download_url, "mp4_time": mp4_time,
                        "mp4_background_image_url": mp4_background_image_url,
                        "mp4_txt_brief": mp4_txt_brief}
            mp4_meta_info_dir = os.path.join(mp4_type_store_dir, str(mp4_id))
            if not os.path.exists(mp4_meta_info_dir):
                os.makedirs(mp4_meta_info_dir)
            with open(os.path.join(mp4_meta_info_dir, str(mp4_id) + ".txt"), "w", encoding="utf-8") as mp4_meta_wf:
                mp4_meta_wf.write(str(mp4_meta_info_dict))
            yield scrapy.Request(url=mp4_download_url, callback=self.parse_video,
                                 meta={'mp4_meta_info_dir': mp4_meta_info_dir})
            yield scrapy.Request(url=mp4_background_image_url, callback=self.parse_background_image,
                                 meta={'mp4_meta_info_dir': mp4_meta_info_dir})

    def parse_video(self, response):
        mp4_meta_info_dir = response.meta["mp4_meta_info_dir"]
        url = response.url
        file_name = url.split("/")[-1]
        video_local_path = os.path.join(mp4_meta_info_dir, file_name)
        with open(video_local_path, "wb") as f:
            f.write(response.body)

    def parse_background_image(self, response):
        mp4_meta_info_dir = response.meta["mp4_meta_info_dir"]
        url = response.url
        file_name = url.split("/")[-1]
        image_local_path = os.path.join(mp4_meta_info_dir, file_name)
        with open(image_local_path, "wb") as f:
            f.write(response.body)