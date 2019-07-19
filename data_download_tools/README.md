## How to use download tools


### Require

+ python 3+, e.g. python==3.6
+ scrapy


### Running web crawlers

```
cd xinpianchang
python start_MP4_meta_info.py
```

### Detail Configuration

[MP4_meta_info.py](data_download_tools/xinpianchang/xinpianchang/spiders/MP4_meta_info.py)

7~13 lines: All video types required by default.

```python
video_type_dict = {'360VR': 'VR', '4k': '4K', 'Technology': '科技', 'Sport': '运动', 'Timelapse': '延时',
                   'Aerial': '航拍', 'Animals': '动物', 'Sea': '大海', 'Beach': '海滩', 'space': '太空',
                   'stars': '星空', 'City': '城市', 'Business': '商业', 'Underwater': '水下摄影',
                   'Wedding': '婚礼', 'Archival': '档案', 'Backgrounds': '背景', 'Alpha Channel': '透明通道',
                   'Intro': '开场', 'Celebration': '庆典', 'Clouds': '云彩', 'Corporate': '企业',
                   'Explosion': '爆炸', 'Film': '电影镜头', 'Green Screen': '绿幕', 'Military': '军事',
                   'Nature': '自然', 'News': '新闻', 'R3d': 'R3d', 'Romantic': '浪漫', 'Abstract': '抽象'}
```

30~35 lines: DOWNLOAD_DELAY The smaller the data capture speed, the faster

```python
custom_settings = {
    'DOWNLOAD_DELAY': 3.5,
    'DOWNLOAD_TIMEOUT': 180,
    'RANDOMIZE_DOWNLOAD_DELAY': True,
    'JOBDIR': "reamin/MP4_meta_info_001"
}
```
