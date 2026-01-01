from calflops import calculate_flops

def format_flops(flops):
    """
    根据 FLOPs 的大小自动添加单位（MFLOPS、GFLOPS、TFLOPS）。
    """
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPS"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPS"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPS"
    else:
        return f"{flops} FLOPS"

def parse_flops(flops_str):
    """解析 FLOPs 字符串并转换为数值（FLOPs 单位）"""
    num, unit = flops_str.split()  # 拆分数值和单位
    num = float(num)

    # 处理单位转换
    if "GFLOPS" in unit:
        num *= 1e9
    elif "MFLOPS" in unit:
        num *= 1e6
    elif "KFLOPS" in unit:
        num *= 1e3
    return num



def analyze_model_flops(net, input_shape, all_data_len, logger=None):
    """
    计算模型的 FLOPs、参数量和总 FLOPs（根据样本数），并返回日志字符串

    Args:
        net: 要分析的模型（已构建好）
        input_shape: 输入的形状 (B, H, W, C)，B建议设置为1
        all_data_len: 数据总样本数（用于计算总 FLOPs）
        logger: 可选 logger 对象（用于输出日志）

    Returns:
        log_msg: 格式化后的日志信息字符串
    """
    net.eval()
    flops_per_sample, macs_per_sample, para = calculate_flops(
        model=net,
        input_shape=input_shape
    )

    flops_per_sample_value = parse_flops(flops_per_sample)
    total_flops = flops_per_sample_value * all_data_len
    formatted_total_flops = format_flops(total_flops)

    log_msg = (f"Model Parameters: {para}, "
               f"FLOPs per sample: {flops_per_sample}, "
               f"Total FLOPs for all samples: {formatted_total_flops}")

    if logger:
        logger.info(log_msg)

    return log_msg
