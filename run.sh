
# python data_processing.py 
python main.py \
      --output_len  24
python main.py \
      --output_len  48
python main.py \
      --output_len  96
python main.py \
      --output_len  192
    # --device 0 \
    # --is_pretrain 0 \
    # --is_finetune 0 \
    # --is_linear_probe 1 \
    # --explain 0 \
    # --test True \
    # --predcit True \
    # --project_name 'power_battery' \
    # --dset_pretrain 'Power-Battery' \
    # --dset_finetune 'Power-Battery' \
    # --data_path './data/local_data_structure' \
    # --batch_size 512 \
    # --num_workers 0 \
    # --scale True \
    # --revin 0 \
    # --n_layers 4 \
    # --n_layers_dec 5 \
    # --prior_dim 6 \
    # --n_heads 16 \
    # --d_model 512 \
    # --dropout 0.15 \
    # --head_dropout 0.05 \
    # --input_len 96 \
    # --output_len 24 \
    # --patch_len 1000 \
    # --stride_ratio 1.0 \
    # --mask_ratio 0.4 \
    # --recon_weight 0.5 \
    # --kl_temperature 0.1 \
    # --n_epochs_pretrain 300 \
    # --lr 1e-4 \
    # --n_epochs_finetune 200 \
    # --head_epochs_ratio 0.2 \
    # --pretrained_model_id 1 \
    # --model_type 'based_model' \
    # --finetuned_model_id 1 \
    # --dist False

# python main.py \
#     --device 3 \
#     --is_pretrain 0 \
#     --is_finetune 0 \
#     --is_linear_probe 1 \
#     --explain 0 \
#     --test True \
#     --predcit True \
#     --project_name 'power_battery' \
#     --dset_pretrain 'Power-Battery' \
#     --dset_finetune 'Power-Battery' \
#     --data_path './data/local_data_structure' \
#     --batch_size 16 \
#     --num_workers 0 \
#     --scale True \
#     --revin 0 \
#     --n_layers 4 \
#     --n_layers_dec 5 \
#     --prior_dim 6 \
#     --n_heads 16 \
#     --d_model 512 \
#     --dropout 0.15 \
#     --head_dropout 0.05 \
#     --input_len 96 \
#     --output_len 48 \
#     --patch_len 1000 \
#     --stride_ratio 1.0 \
#     --mask_ratio 0.4 \
#     --recon_weight 0.5 \
#     --kl_temperature 0.1 \
#     --n_epochs_pretrain 300 \
#     --lr 1e-4 \
#     --n_epochs_finetune 200 \
#     --head_epochs_ratio 0.2 \
#     --pretrained_model_id 1 \
#     --model_type 'based_model' \
#     --finetuned_model_id 1 \
#     --dist False


# python main.py \
#     --device 3 \
#     --is_pretrain 0 \
#     --is_finetune 0 \
#     --is_linear_probe 1 \
#     --explain 0 \
#     --test True \
#     --predcit True \
#     --project_name 'power_battery' \
#     --dset_pretrain 'Power-Battery' \
#     --dset_finetune 'Power-Battery' \
#     --data_path './data/local_data_structure' \
#     --batch_size 16 \
#     --num_workers 0 \
#     --scale True \
#     --revin 0 \
#     --n_layers 4 \
#     --n_layers_dec 5 \
#     --prior_dim 6 \
#     --n_heads 16 \
#     --d_model 512 \
#     --dropout 0.15 \
#     --head_dropout 0.05 \
#     --input_len 96 \
#     --output_len 96 \
#     --patch_len 1000 \
#     --stride_ratio 1.0 \
#     --mask_ratio 0.4 \
#     --recon_weight 0.5 \
#     --kl_temperature 0.1 \
#     --n_epochs_pretrain 300 \
#     --lr 1e-4 \
#     --n_epochs_finetune 200 \
#     --head_epochs_ratio 0.2 \
#     --pretrained_model_id 1 \
#     --model_type 'based_model' \
#     --finetuned_model_id 1 \
#     --dist False



# python main.py \
#     --device 2 \
#     --is_pretrain 0 \
#     --is_finetune 0 \
#     --is_linear_probe 1 \
#     --explain 0 \
#     --test True \
#     --predcit True \
#     --project_name 'power_battery' \
#     --dset_pretrain 'Power-Battery' \
#     --dset_finetune 'Power-Battery' \
#     --data_path './data/local_data_structure' \
#     --batch_size 16 \
#     --num_workers 0 \
#     --scale True \
#     --revin 0 \
#     --n_layers 4 
#     --n_layers_dec 1 \
#     --prior_dim 6 \
#     --n_heads 16 \
#     --d_model 512 \
#     --dropout 0.15 \
#     --head_dropout 0.05 \
#     --input_len 96 \
#     --output_len 192 \
#     --patch_len 1000 \
#     --stride_ratio 1.0 \
#     --mask_ratio 0.4 \
#     --recon_weight 0.5 \
#     --kl_temperature 0.1 \
#     --n_epochs_pretrain 300 \
#     --lr 1e-4 \
#     --n_epochs_finetune 200 \
#     --head_epochs_ratio 0.2 \
#     --pretrained_model_id 1 \
#     --model_type 'based_model' \
#     --finetuned_model_id 1 \
#     --dist False