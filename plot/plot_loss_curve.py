import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(epoch_list,loss_list,save_path=None,title='dataset_name'):
    plt.figure(figsize=(10,6))
    plt.plot(epoch_list,loss_list,marker='o',label='Traing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")

    plt.show()


def plot_average_loss_over_seeds(all_loss_lists, save_path=None, dataset_name='Dataset', show_std=False):
    """
    画出多次训练（多随机种子）下的平均 Loss 曲线，并带标准差区域（可选）

    Parameters:
    - all_loss_lists: List[List[float]]，每次训练的 loss 曲线
    - save_path: str or None，保存图片路径
    - dataset_name: str，图标题中显示的数据集名称
    - show_std: bool，是否显示标准差阴影区域
    """
    all_loss_array = np.array(all_loss_lists)  # shape: [num_seeds, num_epochs]
    mean_loss = np.mean(all_loss_array, axis=0)
    std_loss = np.std(all_loss_array, axis=0)
    epochs = list(range(len(mean_loss)))

    plt.figure(figsize=(10, 6))

    # 主曲线
    plt.plot(epochs, mean_loss, label='Mean Training Loss', color='blue')

    # 阴影区域
    if show_std:
        plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                         color='blue', alpha=0.3, label='±1 std dev')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.title(f"{dataset_name}: Average Training Loss over Seeds")
    plt.title(dataset_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[✔] Saved average loss plot to: {save_path}")

    plt.show()
