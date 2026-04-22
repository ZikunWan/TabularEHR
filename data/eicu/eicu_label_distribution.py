import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_distribution_plots(output_csv="eicu_label_distribution.csv", output_img="eicu_label_distribution.png"):
    base_dir = "/home/ma-user/sfs_turbo/sai6/zkwan/eicu-crd/processed"
    files = {
        "Train": os.path.join(base_dir, "sample_info_train.json"),
        "Val": os.path.join(base_dir, "sample_info_val.json"),
        "Test": os.path.join(base_dir, "sample_info_test.json")
    }
    
    all_data = []
    
    # 1. 加载数据
    for split_name, path in files.items():
        if os.path.exists(path):
            print(f"正在加载 {split_name} 数据: {path} ...")
            with open(path, 'r') as f:
                data = json.load(f)
                for row in data:
                    row['file_split'] = split_name
                all_data.extend(data)
        else:
            print(f"[警告] 找不到文件: {path}")
            
    if not all_data:
        print("未加载到任何数据，请检查路径。")
        return

    # 2. 转换为 DataFrame
    df = pd.DataFrame(all_data)
    results = []
    
    tasks = sorted(df['task_name'].dropna().unique())
    splits_order = ["Train", "Val", "Test"]
    
    # 3. 计算统计数据
    print("\n正在计算分布...")
    for task in tasks:
        task_df = df[df['task_name'] == task]
        for split in splits_order:
            split_df = task_df[task_df['file_split'] == split]
            if split_df.empty:
                continue
            total = len(split_df)
            label_counts = split_df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                percentage = (count / total) * 100
                results.append({
                    "Task": task,
                    "Split": split,
                    "Label": str(label), # 转为字符串方便绘图作为类别
                    "Count": count,
                    "Percentage": percentage
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ CSV 统计结果已保存至: {os.path.abspath(output_csv)}")

    # 4. 绘制直观图表
    print("\n正在生成可视化图表...")
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    # 计算需要多少行和列 (假设每行画 3 个任务的图)
    n_tasks = len(tasks)
    cols = 3
    rows = (n_tasks + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if n_tasks > 1 else [axes]
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_data = results_df[results_df['Task'] == task]
        
        # 使用 seaborn 绘制分组柱状图
        # x轴为划分(Train/Val/Test)，y轴为百分比，颜色代表标签
        sns.barplot(
            data=task_data, 
            x="Split", 
            y="Percentage", 
            hue="Label", 
            ax=ax,
            order=splits_order,
            palette="Set2"
        )
        
        ax.set_title(f"Task: {task}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylim(0, 105) # y轴固定为0-100%
        
        # 在柱子上标注具体数值
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height) and height > 0:
                ax.annotate(f'{height:.1f}%', 
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', 
                            fontsize=10, color='black', xytext=(0, 2), 
                            textcoords='offset points')
                
        # 将图例移到合适的位置
        if ax.get_legend() is not None:
            ax.legend(title='Label', loc='upper right')

    # 隐藏多余的空白子图
    for i in range(n_tasks, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化图表已成功保存至: {os.path.abspath(output_img)}")

if __name__ == "__main__":
    generate_distribution_plots("eicu_label_distribution.csv", "eicu_label_distribution.png")