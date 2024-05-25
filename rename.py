import os

wjj = 'D:\ocean big data\REDATA\\test\palsar\sat'
for n,name in enumerate(os.listdir(wjj)):
    src = wjj + "/" + name
    dst = wjj + "/" + name[0:5] + '.png'
    os.rename(src, dst)