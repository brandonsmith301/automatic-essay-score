# for model_id in model1
# do
#   python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 0
#   python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 1
#   python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 2
#   python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 3
#   python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 4
# done

for model_id in model2
do
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 0
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 1
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 2
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 3
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 4
done

for model_id in model3
do
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 0
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 1
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 2
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 3
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 4
done

for model_id in model4
do
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 0
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 1
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 2
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 3
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 4
done

for model_id in model5
do
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 0
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 1
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 2
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 3
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --fold 4
done