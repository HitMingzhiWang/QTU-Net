import glob


def write_name():
    #npz文件路径
    files = glob.glob(r'datasets/WaterBodies/test_vol_h5/*.npz')
    #txt文件路径
    f = open(r'lists/lists_WaterBodies/test_vol.txt', 'w')
    for i in files:
        name = i.split('\\')[-1]
        name = name[:-4]+'\n'
        f.write(name)

write_name()
