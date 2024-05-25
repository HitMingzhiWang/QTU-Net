import os
import cv2

path1 = 'D:\\paper\\Quantinion\\archive\\Water Bodies Dataset\Masks'

for path in os.listdir(path1):
    img_path = os.path.join(path1,path)
    img = cv2.imread(img_path)
    print(img.shape)
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    print(img.shape)
    # img = cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(img_path,img)