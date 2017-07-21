import os
import glob
# d={}
# for files in glob.glob('/home/zzq/research/windows_file/IIIT-CFW1.0/tmp/*.jpg'):
#     filepath, filename = os.path.split(files)
#     # print filename
#     l=filename.split('.')
#     # print l[0]
#     my_namee=filter(str.isalpha, l[0])
#     print my_namee
#     if d.has_key(my_namee):
#         d[my_namee]+=1
#     else:
#         d[my_namee]=1
# print d
dest='/home/zzq/research/windows_file/IIIT-CFW1.0/dest/'
name={}
for files in glob.glob('/home/zzq/research/windows_file/IIIT-CFW1.0/realFaces/*.jpg'):
    filepath, filename = os.path.split(files)
    l=filename.split('.')
    my_name=filter(str.isalpha,l[0])
    if name.has_key(my_name):
        name[my_name]+=1
    else:
        name[my_name]=1


