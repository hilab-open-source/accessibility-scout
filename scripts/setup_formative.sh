#! /bin/bash

raws_dir=$1
user_model_text=$2
save_dir=$3

python make_environments.py --env-dir "${raws_dir}" --save-dir "${save_dir}/envs"
python make_baselines.py --env-dir "${save_dir}/envs" --save-dir "${save_dir}/baselines"
python make_user_model.py --user-model-text "${user_model_text}" --save-path "${save_dir}/user_model.json"
cp "${save_dir}/user_model.json" "${save_dir}/user_model_base.json"
python predict_baselines.py --baseline-dir "${save_dir}/training_baselines" --user-model-path "data/user_model_empty.json" --save-dir "${save_dir}/predicted_generic"
