# export CUDA_VISIBLE_DEVICES=1

export PYTHONPATH=$PYTHONPATH:./Time-Series-Library

model_name=TABE
model_id=TABE_BTC_r25


python -u run.py \
--task_name long_term_forecast --is_training 0 \
--model TABE --model_id TABE_v0.2 \
--e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --batch_size 10 \
--seq_len 32 --label_len 32 --pred_len 1 --inverse \
--itr 1 --train_epochs 3 --learning_rate 0.001 --des 'Exp' --loss 'MSE' \
--data TABE --features MS --freq d --root_path ./ --data_path dataset_BTC_r25.csv


