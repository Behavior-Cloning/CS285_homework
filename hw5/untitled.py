import sys
from you_get import common as you_get       #导入you-get库

directory = r'E:\bilibili'                   #设置下载目录
url = 'https://www.bilibili.com/video/BV1WU4y1A7jM?spm_id_from=333.999.0.0'      #需要下载的视频地址
sys.argv = ['you-get','-o',directory,url]       #sys传递参数执行下载，就像在命令行一样；‘-o’后面跟保存目录。
you_get.main()