import argparse

# MambaHSI
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_set_path', type=str, default='./data', help='数据集路径')
    parser.add_argument('--work_dir', type=str, default='./', help='工作目录')
    parser.add_argument('--lr', type=float, default=0.0003, help='学习率') # 0.0003 0.0002 0.0001 0.0004
    parser.add_argument('--max_epoch', type=int, default=400, help='最大迭代次数')
    parser.add_argument('--train_samples', type=int, default=30, help='训练样本数')
    parser.add_argument('--val_samples', type=int, default=10, help='验证样本数')
    parser.add_argument('--train_ratio', type=int, default=0.1, help='训练样本比例')
    parser.add_argument('--val_ratio', type=int, default=0.05, help='验证样本比例')
    parser.add_argument('--exp_name', type=str, default='run', help='实验名称')
    parser.add_argument('--record_computecost',type=bool,default=True)
    parser.add_argument("--batch_size", type=int, default=512, help='批次大小')
    parser.add_argument("--net_name", type=str, default="skip_group", help='模型名字')
    parser.add_argument("--log_num",type=int,default=0, help='第几个log文件')
    parser.add_argument('--dataset_index', type=int, default=1,help='数据集索引') # data_set_name_list = ['UP', 'HanChuan', 'HongHu', 'Houston','IP','SA','LongKou']
    parser.add_argument("--flag", type=int, default=1, help='0则ratio，1则samples') # ip 5 pu 5 hc 6 hs 5
    parser.add_argument("--device",type=int,default=1, help='选择GPU')
    parser.add_argument("--patch_length",type=int,default=6, help='patch块大小/2')
    parser.add_argument("--mamba_type",type=str,default="both", help='选择的模块')
    parser.add_argument("--depth",type=int,default=1, help='transformer层数')
    parser.add_argument("--trans",type=bool,default=True, help='是否加上trans模块')
    parser.add_argument("--hidden_dim",type=int,default=64, help='hidden_num')
    parser.add_argument("--heads",type=int,default=8, help='注意力头数')
    parser.add_argument("--use_att",type=bool,default=True, help='双分支融合策略')
    parser.add_argument("--explain", type=str, default="", help='解释说明')

    return parser.parse_args()
# 48 64 80