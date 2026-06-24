#!/bin/bash
set -euo pipefail

python preprocess/Renji/4_generate_death_tte_index.py \
    --root-dir "/data/EHR_data_public/Renji" \
    --output-dir "data/renji_tte_index" \
    --horizon-days 1825
