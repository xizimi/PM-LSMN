import os
import random

if __name__ == '__main__':
    # if not (os.path.exists('img_train') and os.path.exists('mask_train')):
    #     raise FileNotFoundError("请放置数据集文件夹!")
    file2 = 'D:\\BaiduNetdiskDownload\data/index_div/'
    file1 = 'D:\\BaiduNetdiskDownload\data/new_div/'
    image_list = os.listdir(file1)
    mask_list = os.listdir(file2)

    assert len(image_list) == len(mask_list), "数据和标签数量不一致!"

    img_mask = list(zip(image_list, mask_list))
    random.shuffle(img_mask)

    with open("train_gid.txt", 'w') as f:
        for image, mask in img_mask[:int(0.8 * len(img_mask))]:
            f.writelines(file1+'%s '%image+ file2+'%s\n'%mask)
    f.close()

    with open("val_gid.txt", 'w') as f:
        for image, mask in img_mask[int(0.8 * len(img_mask)):]:
            f.writelines(file1+'%s '%image+ file2+'%s\n'%mask)
    f.close()

