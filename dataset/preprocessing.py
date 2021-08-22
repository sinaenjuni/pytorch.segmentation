from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


IMGPATH = "../data/CelebAMask-HQ/CelebA-HQ-img"
SEGPATH = "../data/CelebAMask-HQ/CelebAMask-HQ-mask-anno"


classes = { 'back_ground':0,
            'skin':1,
            'cloth':2,
            'neck':3,
            'neck_l':4,
            'l_ear':5,
            'r_ear':6,
            'ear_r':7,
            'l_brow':8,
            'r_brow':9,
            'l_eye':10,
            'r_eye':11,
            'nose':12,
            'mouth':13,
            'u_lip':14,
            'l_lip':15,
            'eye_g':16,
            'hair':17,
            'hat':18}


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


def getConbineSeg(seg_list):
    ret = np.zeros((512,512,19))
    for i in seg_list:
        seg_type = '_'.join(i.name.split('_')[1:]).replace('.png', '')
        seg_class = classes[seg_type]
        seg = Image.open(i).convert("L")
        # seg = np.array(seg)
        # print(np.unique(seg))
        arr = np.array(seg) // 255 * seg_class
        ret[..., seg_class] = arr
    ret = np.argmax(ret, 2).astype(np.uint8)
    return ret

def saveImg(path:Path, img):
    if not path.parent.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
    Image.fromarray(img).save(path)


if __name__ == "__main__":
    img_list = getImglist(IMGPATH)
    # seg_list = getSeglist(SEGPATH, '24916')
    # seg_list = getSeglist(SEGPATH, '26')
    # cseg = getConbineSeg(seg_list)
    # plt.imshow(cseg)
    # plt.show()

    for i in range(len(img_list)):
        print(i)
        img_id = img_list[i].name.replace('.jpg', '')
        # img = getIMG(img_list[i])
        # showIMG(img)

        seg_list = getSeglist(SEGPATH, img_id)
        cseg = getConbineSeg(seg_list)

        saveImg(Path(f'./CelebA-HQ-mask/{img_id}.png'), cseg)

        # print(np.unique(cseg))
        # plt.imshow(cseg, cmap='gray')
        # plt.show()

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