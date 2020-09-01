import os

listVideos = []

base_folder = './test' 

for vidType in os.listdir(base_folder):
    type_folder = base_folder+'/'+vidType
    for vidRes in os.listdir(type_folder):
        res_folder = type_folder+'/'+vidRes
        for vid in os.listdir(res_folder):
            listVideos.append(res_folder+'/'+vid)

with open('listVideo.csv','w') as f:
    for vid in listVideos[:len(listVideos)-1]:
        name_ext = vid.split('/')[len(vid.split('/'))-1]
        name = name_ext[:len(name_ext)-4]
        f.write(vid+','+name+'\n')
    vid = listVideos[len(listVideos)-1]
    name_ext = vid.split('/')[len(vid.split('/'))-1]
    name = name_ext[:len(name_ext)-4]
    f.write(vid+','+name)
