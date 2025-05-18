import os
from datetime import datetime

def visualize_tensors(ori_x, x, path: str):
    import matplotlib.pyplot as plt
    
    # 获取维度信息
    _, _, n_features = ori_x.shape
    
    # 创建保存结果的文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_folder = os.path.join(path, f'visualization_{timestamp}')
    os.makedirs(result_folder, exist_ok=True)
    
    # 遍历所有特征维度
    for feature_idx in range(n_features):
        # 将数据移到 CPU 并转换为 numpy 数组
        ori_data = ori_x[0, :, feature_idx].cpu().detach().numpy()
        processed_data = x[0, :, feature_idx].cpu().detach().numpy()
        
        # 创建时间步索引
        time_steps = range(len(ori_data))
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 绘制原始数据
        plt.plot(time_steps, ori_data, label='Original Data', color='blue')
        
        # 绘制处理后的数据
        plt.plot(time_steps, processed_data, label='Predicted Data', color='red')
        
        # 添加标题和标签
        plt.title(f'Comparison of Original and Predicted Data - Feature {feature_idx + 1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        result_path = os.path.join(result_folder, f'feature_{feature_idx + 1}.png')
        plt.savefig(result_path)
        plt.close()
        
    return result_folder