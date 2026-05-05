import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def getCAM2(features, img, img_Z, idx):
    processor = ImageProcessor()
    img_Z = processor.reverse_process(img_Z)
    save_path =  '/home/admz/Documents/01/heatmap/osmtrack/03'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cropsave_path = '/home/admz/Documents/01/heatmap/osmtrack/03_crop/'
    if not os.path.exists(cropsave_path):
        os.makedirs(cropsave_path)
    crop_z_save_path = '/home/admz/Documents/01/heatmap/osmtrack/03_crop_Z/'
    if not os.path.exists(crop_z_save_path):
        os.makedirs(crop_z_save_path)
    # os.path = /home/tcm/PycharmProjects/siamft/pysot_toolkit/trackers
    #img = img_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    features = features.to("cpu")
    features = features.squeeze(1).detach().numpy()
    img = cv2.resize(img, (256, 256))
    img = img
    img = np.array(img, dtype=np.uint8)
    cropsave_path = cropsave_path+ str(idx) + '.png'
    crop_Z_save_path = crop_z_save_path+ str(idx) + '.png'
    # mask = features.sum(dim=0, keepdims=False)
    plt.imsave(cropsave_path, img)
    plt.imsave(crop_Z_save_path, img_Z)
    mask = features
    # mask = mask.detach().cpu().numpy()
    mask = mask.transpose((1, 2, 0))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = cv2.resize(mask, (256,256))
    mask = 255 * mask
    mask = mask.astype(np.uint8)
    heatmap = cv2.applyColorMap(255-mask, cv2.COLORMAP_JET)

    img = cv2.addWeighted(src1=img, alpha=0.6, src2=heatmap, beta=0.4, gamma=0)
    name = '/attn_%d.png' % idx
    cv2.imwrite('/home/admz/Documents/01/heatmap/osmtrack/03' + name, img)


def pltshow(pred_map, name):
    import matplotlib.pyplot as plt
    pred_map = pred_map.squeeze(0).cpu()
    pred_map = pred_map.numpy()
    plt.figure(2)
    pred_frame = plt.gca()
    plt.imshow(pred_map, 'jet')
    pred_frame.axes.get_yaxis().set_visible(False)
    pred_frame.axes.get_xaxis().set_visible(False)
    pred_frame.spines['top'].set_visible(False)
    pred_frame.spines['bottom'].set_visible(False)
    pred_frame.spines['left'].set_visible(False)
    pred_frame.spines['right'].set_visible(False)
    pred_name = '/home/admz/Documents/01/heatmap/backbone/18/' + str(name) + '.png'
    #pred_name = '/home/admz/Documents/01/heatmap/ostrack/01/attn_1.png'
    plt.savefig(pred_name, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()


class ImageProcessor:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def reverse_process(self, img_tensor_norm):
        # 反标准化
        img_tensor = img_tensor_norm * self.std + self.mean
        img_tensor = img_tensor.squeeze(0)  # 去除 batch 维度
        img_tensor = img_tensor.permute((1, 2, 0))  # 改变维度顺序，从 (C, H, W) 到 (H, W, C)
        img_arr = img_tensor.cpu().numpy()  # 转换为 NumPy 数组
        img_arr = (img_arr * 255.0).astype(np.uint8)  # 缩放回 [0, 255] 并转换为无符号 8 位整数
        return img_arr


if __name__ == '__main__':
    # feature = torch.rand(1, 16, 16)
    # img_path = '/home/admz/projects/datasets/ODinMJ/20/img/00001.JPG'  # 确保这是正确的文件路径
    # img = cv2.imread(img_path)
    # getCAM2(feature, img, 4)
    pred_map = np.random.rand(256, 256)
    pltshow(pred_map,2)
