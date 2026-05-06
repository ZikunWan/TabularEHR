python preprocess/mimic_iv/5_task_sample_info_gen.py \
  --patient_ids_path /data/zikun_workspace/mimic-iv-3.1_tabular/patient_data/train.csv \
  --output_path /data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train \
  --task next_token_prediction \
  --include_first_admission_pretraining