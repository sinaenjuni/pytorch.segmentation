from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


IMGPATH = "../data/CelebAMask-HQ/CelebA-HQ-img"
SEGPATH = "../data/CelebAMask-HQ/CelebAMask-HQ-mask-anno"

INDEX = {'skin':1,
         ''}

def getImglist(path):
    path = Path(path)
    datalist = list(path.glob('*.jpg'))
    # print(datalist)
    return datalist

def getSeglist(path, idx):
    idx = idx.zfill(5)
    path = Path(path)
    datalist = list(path.glob((f'*/{idx}*.png')))
    print(idx, len(datalist))
    # assert len(datalist) == 12
    return datalist

def getIMG(path):
    return Image.open(path)

def showIMG(img):
    plt.imshow(img)
    plt.show()

def getSeg(path_list):
    ret = 0
    for i, path in enumerate(path_list):
        img = Image.open(path)
        img = np.array(img)//255 * i
        # print(np.unique(img))
        ret += np.array(img)
    print(np.unique(ret))
    # plt.imshow(ret, cmap='jet')
    # plt.show()


if __name__ == "__main__":
    img_list = getImglist(IMGPATH)
    # print(img_list)
    # for i in range(30000):
    #     img_id = img_list[i].name.replace('.jpg', '')
    #     seg_list = getSeglist(SEGPATH, img_id)
    seg_list = getSeglist(SEGPATH, '24916')
    for i in seg_list:
        print(i)

    # img = getIMG(img_list[0]).resize((512, 512))
    # showIMG(img)
    # print(np.array(img.resize((512, 512))).shape)
    # for i in seg_list:
    #     # print(i)
    #     seg = getIMG(i)
    #     # print(np.array(seg).shape)
    #     filtered_img = np.array(img) * (np.array(getIMG(i))//255)
    #     plt.imshow(filtered_img)
    #     plt.show()

    # print(img_list[0].name)
    # for img in img_list[:1]:
    #     seg_list = getSeglist(SEGPATH, img.name.replace('.jpg', ''))
    #     for i in seg_list:
    #         print(i)
    #         showIMG(i)
    # getSeg(seg_list)

    # getSeglist(SEGPATH)