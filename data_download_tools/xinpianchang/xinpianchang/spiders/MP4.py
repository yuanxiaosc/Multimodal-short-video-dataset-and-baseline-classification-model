# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider, Rule
import json
import os

def read_raw_mp4_info_json_file_2_list(mp4_info_file="mp4_info.json"):
    MP4_info_list = list()
    with open(mp4_info_file, 'r', encoding='utf-8') as rf:
        while True:
            line = rf.readline()
            if line:
                line_dict = json.loads(line)
                MP4_info_list.append(line_dict["mp4_download_url"])
            else:
                break
    return MP4_info_list


class Mp4Spider(CrawlSpider):
    name = 'MP4'
    allowed_domains = ['xinpianchang.com']
    start_urls = ['https://www.xinpianchang.com/']
    custom_settings = {
        'DOWNLOAD_DELAY': 2,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
    }

    def start_requests(self):
        self.MP4_base_dir = "MP4_download"
        task_mp4_type_list = ["4k_url", ]
        for mp4_type in task_mp4_type_list:
            mp4_type_store_dir = os.path.join(self.MP4_base_dir, mp4_type)
            if not os.path.exists(mp4_type_store_dir):
                os.makedirs(mp4_type_store_dir)
            MP4_info_list = read_raw_mp4_info_json_file_2_list(os.path.join("task_file", mp4_type + ".json"))
            for url in MP4_info_list:
                yield scrapy.Request(url=url, callback=self.parse_video, meta={'mp4_type': mp4_type},
                                     headers={'Referer': 'https://www.xinpianchang.com/'})

    def parse_video(self, response):
        meta = response.meta
        url = response.url
        mp4_type = response.meta["mp4_type"]
        file_name = url.split("/")[-1]
        mp4_type_store_dir = os.path.join(self.MP4_base_dir, mp4_type)
        video_local_path = os.path.join(mp4_type_store_dir, file_name)
        with open(video_local_path, "wb") as f:
            f.write(response.body)
        yield meta