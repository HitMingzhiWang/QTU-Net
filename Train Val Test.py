import os
import shutil

import cv2

path1 = 'D:\\paper\\Quantinion\\WHDLD\\WHDLD\\ImagesPNG'
pathTrain = 'D:\\paper\\Quantinion\\WHDLD\\WHDLD\\train\\gt'
pathVal = 'D:\\paper\\Quantinion\\WHDLD\\WHDLD\\val\\gt'
pathTest = 'D:\\paper\\Quantinion\\WHDLD\\WHDLD\\test\\gt'
count = 1
for path in os.listdir(path1):
    img_path = os.path.join(path1,path)
    if count<=3458:
        shutil.move(img_path,os.path.join(pathTrain,path))
    elif count>3458 and count <= 4446:
        shutil.move(img_path,os.path.join(pathVal,path))
    else:
        shutil.move(img_path,os.path.join(pathTest,path))
    count = count + 1