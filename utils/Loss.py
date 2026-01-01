def head_loss(loss_func,logits,label):
    height, width = logits.shape[2], logits.shape[3]
    center_h = height // 2
    center_w = width // 2
    # 提取中心像素点
    center_pixels = logits[:, :, center_h, center_w]
    center_pixels = center_pixels.unsqueeze(-1).unsqueeze(-1)
    label = label.unsqueeze(-1).unsqueeze(-1)
    loss = loss_func(center_pixels,label)

    return loss

