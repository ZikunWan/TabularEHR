import os 
import pandas as pd
import json
import datetime
import jsonlines
import numpy
import tqdm

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, int):
            return str(obj)
        if isinstance(obj, float):
            return str(obj)
        elif isinstance(obj, numpy.int64):
            return str(obj)
        else:
            return super(MyEncoder, self).default(obj)


def save_item_whole(source, item, index, root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    print(f"Saving {source} data to jsonl...")
    for subject_id in tqdm.tqdm(list(item.keys())):
        patient_dir = os.path.join(root_dir, str(subject_id))

        if not os.path.exists(patient_dir):
            try:
                os.mkdir(patient_dir)
            except:
                pass

        file_path = os.path.join(patient_dir, source + '.jsonl')
        with open(file_path, 'w') as f:
            for ii in item[subject_id]:
                ii = json.dumps(ii, cls=MyEncoder)
                ii = ii.replace('\u00a0', ' ')  # 替换 U+00A0 为空格
                f.write(ii)
                f.write("\n")

def work(csv_path, source, file_name, current_subject_dict):
    chunksize = 1000000
    # pandas 会自动处理 .gz 后缀
    chunks = pd.read_csv(csv_path, chunksize=chunksize)

    for items in tqdm.tqdm(chunks, desc=f"Reading {file_name}"):
        if "subject_id" in items.columns: # 使用 columns 判断更准确
            # 将 DataFrame 转换为字典列表，效率比 iterrows 高
            records = items.to_dict('records')

            for sample in records:
                sample['file_name'] = file_name
                subject_id = sample["subject_id"]

                # 显式转换 subject_id 为 str，防止部分是从 int 读取
                subject_id = str(subject_id)

                if subject_id in current_subject_dict:
                    current_subject_dict[subject_id].append(sample)
                else:
                    current_subject_dict[subject_id] = [sample]
        else:
            # 如果没有 subject_id 列，跳过该文件
            print(f"Skipping {file_name}: No subject_id column found.")
            break

# 主程序
if __name__ == "__main__":

    root_path = "/home/ma-user/sfs_turbo/sai6/yangqian/tmp_input/mimic-iv-3.1"
    output_path = "/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/patients"
    os.makedirs(output_path, exist_ok=True)

    outside_files = {
        "hosp": [
        "   d_hcpcs.csv.gz", "d_icd_diagnoses.csv.gz", "d_icd_procedures.csv.gz", 
            "d_labitems.csv.gz", "poe_detail.csv.gz", "emar_detail.csv.gz", 
            "provider.csv.gz", "d_hcpcs_unzipped.csv", "d_icd_diagnoses_unzipped.csv",
            "d_icd_procedures_unzipped.csv", "d_labitems_unzipped.csv", 
            "poe_unzipped.csv", "prescriptions_unzipped.csv"
        ],
        "ed": [],
        "icu": [
            "caregiver.csv.gz", "d_items.csv.gz", 
            "d_items_unzipped.csv", "d_items_unzipped_unzipped.csv_gz"
        ],
        "note": ["discharge_detail.csv.gz", "radiology_detail.csv.gz", "index.html"]
    }

    DIR_MAP = {
        "hosp": os.path.join(root_path, "hosp"),
        "icu": os.path.join(root_path, "icu"),
        "ed":  os.path.join(root_path, "ed", 'ed'),
        "note": os.path.join(root_path, 'note', 'note')
    }

    sources = ["hosp", "ed", "icu", "note"]

    for source in sources:
        print(f"========== Processing Source: {source} ==========")

        subject_dict = {} 

        source_dir = DIR_MAP[source]

        if not os.path.exists(source_dir):
            print(f"Directory {source_dir} not found, skipping...")
            continue

        csv_list = os.listdir(source_dir)
        index_csv = 0

        for csv_file in csv_list:
             # 1. 过滤非 CSV 文件 (支持 .csv 和 .csv.gz)
            if not (csv_file.endswith(".csv") or csv_file.endswith(".csv.gz")):
                continue

            # 2. 过滤字典表 (以 d_ 开头) 和黑名单文件
            if csv_file.startswith("d_") or (csv_file in outside_files.get(source, [])):
                continue

            # 3. 提取纯文件名 (处理 .csv.gz)
            # admissions.csv.gz -> admissions
            file_name_clean = csv_file.split('.')[0]

            # 构建完整路径
            full_csv_path = os.path.join(source_dir, csv_file)

            print(f"Processing File: {file_name_clean} (Original: {csv_file})")


            work(csv_path=full_csv_path, 
                source=source, 
                file_name=file_name_clean, 
                current_subject_dict=subject_dict)

            index_csv += 1

        # 保存该 source 的所有数据
        save_item_whole(source=source, index=index_csv, item=subject_dict, root_dir=output_path)

        # 显式释放内存
        del subject_dict