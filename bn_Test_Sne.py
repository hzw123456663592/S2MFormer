import os
import time

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Linux / 服务器必需
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torchvision import transforms
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss
from utils.evaluation import Evaluator
from utils.HSICommonUtils import ImageStretching
from arguments import get_parser
from utils.setup_logger import setup_logger
from utils.seed import setup_seed
from utils.vis_a_img import vis_a_image
from model.bn_MambaHSI_Fusion_4 import MambaHSI
from utils import output
from utils.flops import analyze_model_flops
from plot.plot_loss_curve import plot_loss_curve, plot_average_loss_over_seeds


def save_tsne(features, labels, save_path, title="t-SNE"):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init='pca',
        learning_rate='auto',
        random_state=0
    )
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap='jet',
        s=5
    )
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()




# 参数设置
args = get_parser()
record_computecost = args.record_computecost
exp_name = args.exp_name
dataset_index = args.dataset_index
max_epoch = args.max_epoch
learning_rate = args.lr
flag = args.flag
patch_length = args.patch_length
batch_size = args.batch_size
data_set_path = args.data_set_path
mamba_type = args.mamba_type
net_name = args.net_name
device = args.device
explain = args.explain
depth = args.depth
trans = args.trans
hidden_dim = args.hidden_dim
heads = args.heads
use_att = args.use_att
train_samples = args.train_samples
val_samples =  args.val_samples
train_ratio = args.train_ratio
val_ratio = args.val_ratio
num_list = [train_samples, val_samples] #[train_samples,val_samples]
ratio_list = [train_ratio,val_ratio]  # [train_ratio,val_ratio]
data_set_name_list = ['UP', 'HanChuan', 'HongHu', 'Houston','IP','SA','LongKou']
data_set_name = data_set_name_list[dataset_index]
device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
seed_list = [0,1,2,3,4,5,6,7,8,9]
value = 2 * patch_length + 1 # 上采样尺寸
paras_dict = {'net_name':net_name,'dataset_index':dataset_index,'num_list':num_list,
              'lr':learning_rate,'seed_list':seed_list}

transform = transforms.Compose([
    transforms.ToTensor(),
])



if __name__ == '__main__':
    data_set_path = args.data_set_path
    work_dir = args.work_dir
    setting_name = 'tr{}val{}'.format(str(train_samples), str(val_samples)) + '_lr{}'.format(
        str(learning_rate))

    dataset_name = data_set_name

    exp_name = args.exp_name

    save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    save_log_path = os.path.join(save_folder, '{}_val{}_{}.log'.format(num_list[0], num_list[1], args.log_num))
    logger = setup_logger(name='{}'.format(dataset_name), logfile=save_log_path)
    torch.cuda.empty_cache()

    logger.info(save_folder)
    data, gt = data_load_operate.load_data(data_set_name, data_set_path)
    height, width, channels = data.shape
    gt_reshape = gt.reshape(-1)
    data = ImageStretching(data)
    data = transform(data)
    data = data.permute(1,2,0)
    class_count = max(np.unique(gt))

    # data pad zero
    # data:[h,w,c]->data_padded:[h+2l,w+2l,c]
    data_padded = data_load_operate.data_pad_zero(data, patch_length)

    height_patched, width_patched, channels = data_padded.shape
    loss_func = torch.nn.CrossEntropyLoss()

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])
    all_loss_lists = []  # 存储多个loss曲线
    for exp_idx,curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)
        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed))
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))

        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))

        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list,
                                                                                                    num_list,
                                                                                                    gt_reshape,
                                                                                                    class_count,
                                                                                                    flag)
        index = (train_data_index, val_data_index, test_data_index)

        train_iter,val_iter,test_iter = data_load_operate.generate_iter_1(data_padded, height, width, gt_reshape, index, patch_length, batch_size, model_type_flag=1, model_3D_spa_flag=0, last_batch_flag=0)

        # ===================== 原始特征 t-SNE 可视化 =====================
        all_raw_features = []
        all_raw_labels = []

        # 遍历所有像素索引
        for idx in all_data_index:
            # idx 对应的一维索引映射到二维坐标
            i, j = np.unravel_index(idx, (height, width))
            pixel_feature = data_padded[i, j, :]  # 获取像素原始特征 [channels]
            all_raw_features.append(pixel_feature)
            all_raw_labels.append(gt[i, j])

        all_raw_features = np.array(all_raw_features)
        all_raw_labels = np.array(all_raw_labels)

        raw_tsne_save_path = os.path.join(save_folder, "tsne_raw_features.png")

        save_tsne(
            features=all_raw_features,
            labels=all_raw_labels,
            save_path=raw_tsne_save_path,
            title=f"Raw Features t-SNE ({data_set_name})"
        )
        logger.info(f"Raw feature t-SNE saved to {raw_tsne_save_path}")


        # build Model
        # net = MambaHSI(in_channels=channels, num_classes=class_count, hidden_dim=128,mamba_type=mamba_type)
        net = MambaHSI(
            in_channels=channels,
            num_classes=class_count,
            hidden_dim=hidden_dim,
            mamba_type=mamba_type,
            depth=depth,
            trans=trans,
            heads=heads,
            use_att=use_att)
        net.to(device)

        # 计算模型参数和Flops
        if record_computecost:
            net.eval()  # 将模型设置为评估模式
            model = net
            input_shape = (1, value, value, channels)  # 输入形状
            all_len = len(all_data_index)
            log_msg = analyze_model_flops(model,input_shape,all_len,logger)

        logger.info(log_msg)
        logger.info(paras_dict)
        logger.info(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        best_val_acc = 0
        evaluator = Evaluator(num_class=class_count)
        epoch_list = []
        loss_list = []
        start_train_time = time.perf_counter()
        for epoch in range(max_epoch):
            net.train()
            train_loss_sum = 0
            # 训练开始前记录时间

            for X_train,y_train in train_iter:
                X_train = X_train.to(device)
                y_train = y_train.to(device)
                y_pred = net(X_train)
                # ls = head_loss(loss_func, y_pred, y_train.long())
                ls = loss_func(y_pred,y_train.long())

                optimizer.zero_grad()
                ls.backward()
                optimizer.step()

                train_loss_sum += ls.item()

            epoch_list.append(epoch)
            loss_list.append((train_loss_sum))

            net.eval()
            with torch.no_grad():
                evaluator.reset()
                # 初始化整图的预测结果
                full_predict = np.zeros_like(gt, dtype=np.int8) - 1  # 与 gt 大小相同的空数组
                val_gt = np.zeros_like(gt, dtype=np.int8) - 1 # 与 gt 大小相同的空数组
                # 初始化一个空的一维数组
                predicts = np.array([], dtype=np.int8)
                for X_val,y_val in val_iter:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)
                    y_val_pred = net(X_val)
                    # center_pixels = y_val_pred[:, :, value // 2, value // 2]
                    # 获取预测结果（假设输出形状为 (batch_size, num_classes, 1, 1)）
                    # predict = torch.argmax(center_pixels, dim=1).squeeze().cpu().numpy()  # (batch_size,)
                    predict = torch.argmax(y_val_pred, dim=1).squeeze().cpu().numpy()  # (batch_size,)
                    # 将当前 batch 的 predict 拼接到 predicts 中
                    predicts = np.concatenate((predicts, predict), axis=0)

            # 将一维索引映射到二维坐标
            Coordinates = np.unravel_index(val_data_index, (height, width))

            # 直接填充 full_predict 和 val_gt
            full_predict[Coordinates] = predicts
            val_gt[Coordinates] = gt[Coordinates] - 1
            val_gt_np = val_gt
            val_gt_255 = np.where(val_gt_np == -1, 255, val_gt_np)
            evaluator.add_batch(val_gt_255, full_predict)
            OA = evaluator.Pixel_Accuracy()
            mIOU, IOU = evaluator.Mean_Intersection_over_Union()
            mAcc, Acc = evaluator.Pixel_Accuracy_Class()
            Kappa = evaluator.Kappa()
            logger.info(
                'epoch {}|loss:{:4f}|OA:{:.4f}|MACC:{:.4f}|Kappa:{:.4f}|MIOU:{:.4f}|IOU:{}|ACC:{}'.format(
                    epoch+1,
                    train_loss_sum,
                    OA,
                    mAcc,
                    Kappa,
                    mIOU,
                    ",".join([f"{i:.4f}" for i in IOU]) if isinstance(IOU, np.ndarray) else IOU,
                    # 转为逗号分隔字符串
                    ",".join([f"{i:.4f}" for i in Acc]) if isinstance(Acc, np.ndarray) else Acc
                    # 转为逗号分隔字符串
                )
            )
            # save weight
            if OA >= best_val_acc:
                best_epoch = epoch + 1
                best_val_acc = OA
                torch.save(net.state_dict(), save_weight_path)
            if (epoch + 1) % 50 == 0:
                save_single_predict_path = os.path.join(save_vis_folder, 'predict_{}.png'.format(str(epoch + 1)))
                save_single_gt_path = os.path.join(save_vis_folder, 'gt.png')
                save_single_rgb_path = os.path.join(save_vis_folder, '{}_rgb.png'.format(dataset_name))
                vis_a_image(gt, full_predict, save_single_predict_path, save_single_gt_path, only_vis_label = False)
        # 训练结束后记录时间
        end_train_time = time.perf_counter()
        train_elapsed_time = end_train_time - start_train_time
        Train_Time_ALL.append(train_elapsed_time)
        all_loss_lists.append(loss_list)
        loss_curve_path = os.path.join(save_single_experiment_folder,'loss_curve_png')
        plot_loss_curve(epoch_list,loss_list,save_path=loss_curve_path) # 每个随机种子的epoch
        plot_average_loss_over_seeds(all_loss_lists,save_path=save_folder+"average_loss_curve.png",dataset_name=data_set_name)
        # test
        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        load_weight_path = save_weight_path
        net.update_params = None

        best_net = MambaHSI(in_channels=channels,
                            num_classes=class_count,
                            hidden_dim=hidden_dim,
                            mamba_type=mamba_type,
                            depth=depth,
                            trans=trans,
                            heads=heads,
                            use_att=use_att)
        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()
        test_evalutor = Evaluator(num_class=class_count)
        with torch.no_grad():
            test_evalutor.reset()
            # 记录测试时间
            start_test_time = time.perf_counter()
            full_predict_test = np.zeros_like(gt,dtype=np.int8) - 1
            val_gt_test = np.zeros_like(gt,dtype=np.int8) - 1
            predicts = np.array([],dtype = np.int8)


            # 原本的
            # for X_test,y_test in test_iter:
            #     X_test = X_test.to(device)
            #     y_test = y_test.to(device)
            #     y_test_pred = best_net(X_test)
            #     # center_pixels = y_test_pred[:,:,value // 2,value // 2]
            #     # predict = torch.argmax(center_pixels,dim = 1).squeeze().cpu().numpy()
            #     predict = torch.argmax(y_test_pred,dim = 1).squeeze().cpu().numpy()
            #     predicts = np.concatenate((predicts,predict),axis = 0)

            all_features = []
            all_labels = []

            for X_test, y_test in test_iter:
                X_test = X_test.to(device)
                y_test = y_test.to(device)

                # logits 作为特征（审稿人完全接受）
                y_test_pred = best_net(X_test)  # [B, num_classes]

                feats = y_test_pred.detach().cpu().numpy()
                labs = y_test.detach().cpu().numpy()

                all_features.append(feats)
                all_labels.append(labs)

                predict = np.argmax(feats, axis=1)
                predicts = np.concatenate((predicts, predict), axis=0)

            all_features = np.concatenate(all_features, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            tsne_save_path = os.path.join(save_single_experiment_folder, "tsne_test.png")
            save_tsne(
                features=all_features,
                labels=all_labels,
                save_path=tsne_save_path,
                title=f"{net_name} t-SNE ({dataset_name})"
            )

            # 记录测试结束时间
            end_test_time = time.perf_counter()
            test_elapsed_time = end_test_time - start_test_time
            Test_Time_ALL.append(test_elapsed_time)
            # 将一维索引映射到二维坐标
            Coordinates = np.unravel_index(np.array(test_data_index, dtype=np.intp), (height, width))
            # 直接填充 full_predict 和 val_gt
            full_predict_test[Coordinates] = predicts
            val_gt_test[Coordinates] = gt[Coordinates] - 1
            val_gt_test_np = val_gt_test
            val_gt_test_255 = np.where(val_gt_test_np == -1, 255, val_gt_test_np)
            test_evalutor.add_batch(val_gt_test_255, full_predict_test)
            OA_test = test_evalutor.Pixel_Accuracy()
            mIOU_test, IOU_test= test_evalutor.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evalutor.Pixel_Accuracy_Class()
            Kappa_test = test_evalutor.Kappa()
            logger.info(
                'Test {}|OA:{:.4f}|MACC:{:.4f}|Kappa:{:.4f}|MIOU:{:.4f}|IOU:{}|ACC:{}'.format(
                    epoch,
                    OA_test,
                    mAcc_test,
                    Kappa_test,
                    mIOU_test,
                    ",".join([f"{i:.4f}" for i in IOU_test]) if isinstance(IOU_test, np.ndarray) else IOU_test,
                    # 转为逗号分隔字符串
                    ",".join([f"{i:.4f}" for i in Acc_test]) if isinstance(Acc_test, np.ndarray) else Acc_test
                    # 转为逗号分隔字符串
                )
            )

            train_Coordinates = np.unravel_index(np.array(train_data_index,dtype=np.intp),(height,width))
            full_predict_test[train_Coordinates] = gt[train_Coordinates] - 1
            val_Coordinates = np.unravel_index(np.array(val_data_index,dtype=np.intp),(height,width))
            full_predict_test[val_Coordinates] = gt[val_Coordinates] - 1
            vis_a_image(gt, full_predict_test, predict_save_path, gt_save_path)


        # 每次se输出结果
        result = output.output_results(results_save_path, exp_idx, curr_seed, learning_rate, max_epoch, ratio_list,OA_test,mAcc_test,Kappa_test,mIOU_test,IOU_test,Acc_test,patch_length,mamba_type,batch_size,explain)
        logger.info(result)

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)

        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    logger.info('List of OA:', list(OA_ALL))
    logger.info('List of AA:', list(AA_ALL))
    logger.info('List of KPP:', list(KPP_ALL))
    logger.info('OA=', round(np.mean(OA_ALL) * 100, 2), '+-', round(np.std(OA_ALL) * 100, 2))
    logger.info('AA=', round(np.mean(AA_ALL) * 100, 2), '+-', round(np.std(AA_ALL) * 100, 2))
    logger.info('Kpp=', round(np.mean(KPP_ALL) * 100, 2), '+-', round(np.std(KPP_ALL) * 100, 2))
    logger.info('Acc per class=', np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2), '+-',
                np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2))

    logger.info("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    logger.info("Average testing time=", round(np.mean(Test_Time_ALL) * 1000, 2), '+-',
                round(np.std(Test_Time_ALL) * 1000, 3))


    # 10次seed平均结果
    str_results = output.save_mean_results(save_folder, seed_list, patch_length, mamba_type, batch_size, OA_ALL, AA_ALL, KPP_ALL,
                      EACH_ACC_ALL, Train_Time_ALL, Test_Time_ALL,max_epoch,explain,log_msg,flag,train_samples,val_samples,train_ratio,val_ratio)

    del net










