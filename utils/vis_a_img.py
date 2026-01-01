import numpy as np
import spectral as spy
from spectral import spy_colors
import torch
import matplotlib.pyplot as plt

def vis_a_image(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=False):
    visualize_predict(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=only_vis_label)
    visualize_predict(gt_vis,pred_vis,save_single_predict_path.replace('.png','_mask.png'),save_single_gt_path,only_vis_label=True)



def visualize_predict(gt, predict_full, save_predict_path, save_gt_path, only_vis_label=False):
    predict_full = predict_full + 1
    if only_vis_label:
        vis_predict = np.where(gt == 0, gt, predict_full)  # 生成的是mask图像
    else:
        vis_predict = predict_full  # 直接生成
    # spy.save_rgb 的主要功能是将高光谱数据中的三个波段（通常是红、绿、蓝波段）组合成一张 RGB 图像，并将其保存为文件
    spy.save_rgb(save_predict_path, vis_predict, colors=spy_colors)
    spy.save_rgb(save_gt_path, gt, colors=spy_colors)

