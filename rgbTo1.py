import os
import cv2



for path in os.listdir('D:\\paper\\Quantinion\\WHDLD\\WHDLD\\ImagesPNG'):
    img_path = os.path.join('D:\\paper\\Quantinion\\WHDLD\\WHDLD\\ImagesPNG',path)
    img = cv2.imread(img_path)
    for i in range(256):
        for j in range(256):
            if img[i][j][0] == 255 and img[i][j][1] == 0 and img[i][j][2] == 0:
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255
            else:
                img[i][j][0] = 0
                img[i][j][1] = 0
                img[i][j][2] = 0
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    cv2.imwrite(img_path, img)
