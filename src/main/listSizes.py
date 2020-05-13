import os

listVideos = []

base_folder = './test'

for vidType in os.listdir(base_folder):
    type_folder = base_folder+'/'+vidType
    for vidRes in os.listdir(type_folder):
        res_folder = type_folder+'/'+vidRes
        for vid in os.listdir(res_folder):
            listVideos.append(res_folder+'/'+vid)

with open('listVideoSizes.csv','w') as f:
    for vid in listVideos[:len(listVideos)-1]:
        name_ext = vid.split('/')[len(vid.split('/'))-1]
        name = name_ext[:len(name_ext)-4]
        os.system('ls -lrt '+vid+" | awk '{print $5}' > output.txt")
        with open('output.txt','r') as f2:
            size = f2.read()
        f.write(vid+','+name+','+str(size))
    vid = listVideos[len(listVideos)-1]
    name_ext = vid.split('/')[len(vid.split('/'))-1]
    name = name_ext[:len(name_ext)-4]
    os.system('ls -lrt '+vid+" | awk '{print $5}' > output.txt")
    with open('output.txt','r') as f2:
        size = f2.read()
        f.write(vid+','+name+','+str(size)+'\n')

#path = './test1/test4/1.csv'
#cmd ='ls -lrt '+path+" | awk '{print $5}'"
#a = os.system('ls -lrt '+path+" | awk '{print $5}'")
#print(int(a))
