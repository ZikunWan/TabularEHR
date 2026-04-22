"""
生成 patient_ehr 目录

从 EHRSHOT_ASSETS/data/ehrshot.csv 生成 patient_ehr/{patient_id}.csv 文件
"""

import pandas as pd
import json
import os
from tqdm import tqdm

def load_dictionaries():
    """加载所有医疗编码字典"""
    print("\n📚 加载医疗编码字典...")
    
    dictionaries = {}
    
    # 加载主字典
    try:
        with open('evaluation/ehrshot-benchmark-main/EHRSHOT_ASSETS/models/clmbr/token_2_code.json', 'r') as f:
            token_2_code = json.load(f)
        
        with open('evaluation/ehrshot-benchmark-main/EHRSHOT_ASSETS/models/clmbr/token_2_description.json', 'r') as f:
            token_2_description = json.load(f)
        
        # 创建 code -> description 映射
        code_2_description = {}
        code_2_token = {v: k for k, v in token_2_code.items()}
        for code, token in code_2_token.items():
            if token in token_2_description:
                desc = token_2_description[token]
                # 过滤无效描述（重复编码）
                if desc and desc != code and desc != 'None':
                    code_2_description[code] = desc
        
        dictionaries['main'] = code_2_description
        print(f"  ✅ 主字典: {len(code_2_description):,} 条")
    
    except Exception as e:
        print(f"  ⚠️ 主字典加载失败: {e}")
        dictionaries['main'] = {}
    
    # 加载 CPT4 字典
    try:
        with open('evaluation/ehrshot-benchmark-main/EHRSHOT_ASSETS/models/clmbr/cpt4_code.json', 'r') as f:
            dictionaries['cpt4'] = json.load(f)
        print(f"  ✅ CPT4 字典: {len(dictionaries['cpt4']):,} 条")
    except Exception as e:
        print(f"  ⚠️ CPT4 字典加载失败: {e}")
        dictionaries['cpt4'] = {}
    
    # 加载 ICD10PCS 字典
    try:
        with open('evaluation/ehrshot-benchmark-main/EHRSHOT_ASSETS/models/clmbr/icd10pcs.json', 'r') as f:
            dictionaries['icd10pcs'] = json.load(f)
        print(f"  ✅ ICD10PCS 字典: {len(dictionaries['icd10pcs']):,} 条")
    except Exception as e:
        print(f"  ⚠️ ICD10PCS 字典加载失败: {e}")
        dictionaries['icd10pcs'] = {}
    
    return dictionaries

def get_description(code, dictionaries):
    """根据医疗编码获取描述"""
    if pd.isna(code) or not code:
        return ''
    
    code = str(code)
    
    # 首先尝试从主字典查找（包含前缀的完整编码）
    if code in dictionaries['main']:
        return dictionaries['main'][code]
    
    # 如果是 CPT4 编码
    if code.startswith('CPT4/'):
        clean_code = code.split('/', 1)[-1]
        if clean_code in dictionaries['cpt4']:
            return dictionaries['cpt4'][clean_code]
    
    # 如果是 ICD10PCS 编码
    elif code.startswith('ICD10PCS/'):
        clean_code = code.split('/', 1)[-1]
        if clean_code in dictionaries['icd10pcs']:
            return dictionaries['icd10pcs'][clean_code]
    
    # 如果都找不到，返回空字符串
    return ''

def main():
    # 配置路径
    input_file = 'evaluation/ehrshot-benchmark-main/EHRSHOT_ASSETS/data/ehrshot.csv'
    output_dir = 'evaluation/ehrshot-benchmark-main/EHRSHOT_ASSETS/data/patient_ehr'
    failed_rows = []
    print("=" * 60)
    print("生成 patient_ehr 目录")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到文件 {input_file}")
        return
    
    # 加载字典
    dictionaries = load_dictionaries()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 创建输出目录: {output_dir}")
    
    # 读取 ehrshot.csv
    print(f"\n📂 读取 {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"✅ 加载完成: {len(df):,} 条记录")
    
    # 显示列信息
    print(f"\n📋 列名: {list(df.columns)}")
    
    # 按患者分组
    print(f"\n👥 按 patient_id 分组...")
    grouped = df.groupby('patient_id')
    total_patients = len(grouped)
    print(f"✅ 共 {total_patients:,} 个患者")
    
    # 统计描述填充情况
    total_codes = 0
    filled_descriptions = 0
    
    # 生成每个患者的 CSV
    print(f"\n🔄 生成患者 CSV 文件...")
    
    for patient_id, patient_df in tqdm(grouped, desc="处理患者", total=total_patients):
        # 创建副本避免警告
        patient_df = patient_df.copy()
        
        # 删除不需要的列
        columns_to_drop = []
        if 'Unnamed: 0' in patient_df.columns:
            columns_to_drop.append('Unnamed: 0')
        if 'patient_id' in patient_df.columns:
            columns_to_drop.append('patient_id')
        if 'visit_id' in patient_df.columns:
            columns_to_drop.append('visit_id')
        
        if columns_to_drop:
            patient_df = patient_df.drop(columns=columns_to_drop)
        
        # 填充 description 列
        if 'code' in patient_df.columns:
            patient_df['description'] = patient_df['code'].apply(
                lambda x: get_description(x, dictionaries)
            )
            missing_mask = (patient_df['description'] == '') & (patient_df['code'].notna())
            if missing_mask.any():
                missed_data = patient_df[missing_mask].copy()
                missed_data['patient_id'] = patient_id # 记录是哪个患者出的问题
                failed_rows.append(missed_data[['patient_id', 'omop_table', 'code']])
            # 统计
            total_codes += len(patient_df)
            filled_descriptions += (patient_df['description'] != '').sum()
        else:
            patient_df['description'] = ''
        
        # 重新排列列顺序以匹配 EHR-R1 格式
        columns_order = ['omop_table', 'code', 'description', 'start', 'end', 'value', 'unit']
        available_columns = [col for col in columns_order if col in patient_df.columns]
        patient_df = patient_df[available_columns]
        
        # 按时间排序
        if 'start' in patient_df.columns:
            patient_df = patient_df.sort_values('start')
        
        # 保存
        output_file = os.path.join(output_dir, f'{patient_id}.csv')
        patient_df.to_csv(output_file, index=False)
    
    print(f"\n✅ 完成！生成了 {total_patients:,} 个患者 CSV 文件")
    
    # 显示描述填充统计
    if total_codes > 0:
        fill_rate = filled_descriptions / total_codes * 100
        print(f"\n📊 描述填充统计:")
        print(f"   总编码数: {total_codes:,}")
        print(f"   已填充描述: {filled_descriptions:,}")
        print(f"   填充率: {fill_rate:.1f}%")
    if failed_rows:
        all_failed_df = pd.concat(failed_rows)
        
        # 1. 保存明细到文件，方便你手动查看
        report_path = 'missing_descriptions_report.csv'
        all_failed_df.to_csv(report_path, index=False)
        print(f"\n⚠️ 发现 {len(all_failed_df):,} 条记录缺失描述。")
        print(f"📝 详细列表已保存至: {report_path}")

        # 2. 统计出现频率最高的前 10 个缺失编码
        print("\n🔝 出现频率最高的缺失编码 Top 10:")
        top_missing = all_failed_df.groupby(['omop_table', 'code']).size().reset_index(name='count')
        top_missing = top_missing.sort_values(by='count', ascending=False).head(10)
        print(top_missing.to_string(index=False))
    else:
        print("\n🎉 完美！所有编码都找到了描述。")

if __name__ == '__main__':
    main()
