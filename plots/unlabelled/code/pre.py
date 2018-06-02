import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

# root path depends on your computer
root = './9/'
save_root = './resized_9'
resize_size = 28

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + '9'):
    os.mkdir(save_root + '9')

img_list = os.listdir(root)
print(img_list)

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = imresize(img, (resize_size, resize_size))
    plt.imsave(fname=save_root + img_list[i], arr=img)

    if (i % 100) == 0:
        print('%d images complete' % i)