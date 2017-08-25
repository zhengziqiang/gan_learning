#coding=utf-8
import re
import string
import sys
import os
import urllib

# url='http://tieba.baidu.com/p/2521298181'  #网页地址
url='https://www.gwern.net/Danbooru2017'
imagecontent=urllib.urlopen(url).read()#图片内容
urllist=re.findall(r'src="(http.+?\.jpg)"',imagecontent,re.I)
if not urllist:
	print 'not found'
else:
	filepath=os.getcwd()+'/pythonimage'
	if os.path.exists(filepath) is False:
		os.mkdir(filepath)
	x=1
	print u'爬虫'
	for imageurl in urllist:
		temp=filepath+"/%s.jpg"%x
		print u'正在下载第%d张照片'%x
		print imageurl
		urllib.urlretrieve(imageurl,temp)
		x+=1
        print u'图片下载完成'+filepath