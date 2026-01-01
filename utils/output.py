import os
import numpy as np

# Output infors 每一次随机种子的结果
def output_results(results_save_path, exp_idx, curr_seed, learning_rate, max_epoch, ratio_list,OA_test,mAcc_test,Kappa_test,mIOU_test,IOU_test,Acc_test,patch_length,mamba_type,batch_size,explain):
    f = open(results_save_path, 'a+')
    str_results = '\n======================' \
                  + " exp_idx=" + str(exp_idx) \
                  + " seed=" + str(curr_seed) \
                  + " learning_rate=" + str(learning_rate) \
                  + " train_ratio=" + str(ratio_list[0]) \
                  + " val_ratio=" + str(ratio_list[1]) \
                  + ' patch=' + str(2 * patch_length + 1) \
                  + ' type=' + str(mamba_type) \
                  + ' batch_size=' + str(batch_size) \
                  + ' max_epoch=' + str(max_epoch) \
                  + ' explain=' + str(explain) \
                  + " ======================" \
                  + "\nOA=" + str(OA_test) \
                  + "\nAA=" + str(mAcc_test) \
                  + '\nkpp=' + str(Kappa_test) \
                  + '\nmIOU_test:' + str(mIOU_test) \
                  + "\nIOU_test:" + str(IOU_test) \
                  + "\nAcc_test:" + str(Acc_test) + "\n"
    f.write(str_results)
    f.close()
    return str_results
    logger.info(str_results)


# 10次随机种子的平均结果保存函数
def save_mean_results(save_folder, seed_list, patch_length, mamba_type, batch_size,
                      OA_ALL, AA_ALL, KPP_ALL, EACH_ACC_ALL, Train_Time_ALL, Test_Time_ALL,
                      max_epoch, explain, log_msg, Flag, train_samples, val_samples, train_ratio, val_ratio):

    # 创建保存路径
    mean_result_path = os.path.join(save_folder, 'mean_result.txt')

    # Flag = 1 固定样本数；Flag = 0 按比例划分
    if Flag == 1:
        sample_setting_str = f"\ntrain_samples_per_class = {train_samples}, val_samples_per_class = {val_samples}"
    elif Flag == 0:
        sample_setting_str = f"\ntrain_ratio = {train_ratio}, val_ratio = {val_ratio}"
    else:
        sample_setting_str = "\nUnknown sampling setting."

    # 构建结果字符串
    str_results = (
        f'\n\n***************Mean result of {len(seed_list)} times runs ********************'
        f'\npatch = {2 * patch_length + 1}'
        f'\ntype = {mamba_type}'
        f'\nbatch_size = {batch_size}'
        f'\nmax_epoch = {max_epoch}'
        f'\nparam flops totalFlops = {log_msg}'
        f'\nexplain = {explain}'
        f'{sample_setting_str}'  # 添加样本信息
        f'\nList of OA: {list(OA_ALL)}'
        f'\nList of AA: {list(AA_ALL)}'
        f'\nList of KPP: {list(KPP_ALL)}'
        f'\nOA = {round(np.mean(OA_ALL) * 100, 2)} +- {round(np.std(OA_ALL) * 100, 2)}'
        f'\nAA = {round(np.mean(AA_ALL) * 100, 2)} +- {round(np.std(AA_ALL) * 100, 2)}'
        f'\nKpp = {round(np.mean(KPP_ALL) * 100, 2)} +- {round(np.std(KPP_ALL) * 100, 2)}'
        f'\nAcc per class =\n{np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)} +- {np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)}'
        f'\nAverage training time = {np.round(np.mean(Train_Time_ALL), decimals=2)} +- {np.round(np.std(Train_Time_ALL), decimals=3)}'
        f'\nAverage testing time = {np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)} +- {np.round(np.std(Test_Time_ALL) * 100, decimals=3)}'
    )

    # 写入文件
    with open(mean_result_path, 'a+') as f:
        f.write(str_results)
