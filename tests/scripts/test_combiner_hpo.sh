export PYTHONPATH=$PYTHONPATH:./Time-Series-Library


test_name='Cbm_ahpo'
models_used='S_E_TM'


# HPO
etc_desc='yes'

# for interval in 1 3 5 10 15
for interval in 3 5 10 20
do
python -u run.py \
    --model TABE --model_id $test_name'_w_'$models_used'_('$etc_desc')' \
    --task_name long_term_forecast --loss 'MAE' --is_training 1 \
    \
    --data TABE --features MS --freq d --num_workers 1 \
    --target_datatype LogRet --root_path ./ --data_path dataset/btc/BTC_ret_1d.csv \
    \
    --seq_len 32 --label_len 32 --pred_len 1 --inverse \
    --batch_size 16 --lradj type3 --learning_rate 0.001 --dropout 0.1 --itr 1 --train_epochs 3  \
    --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 \
    \
    --basemodel 'SarimaModel' \
    --basemodel 'EtsModel' \
    --basemodel 'TimeMoE' \
    --basemodel 'CMamba --d_model 128 --d_ff 128 --head_dropout 0.1 --channel_mixup --gddmlp --sigma 1.0 --pscan --avg --max --reduction 2' \
    --basemodel 'iTransformer' \
    --basemodel 'DLinear' \
    --basemodel 'PatchTST' \
    --basemodel 'TimeXer' \
    --combiner '--adaptive_hpo --hpo_interval '$interval' --max_hpo_eval 300 --patience 10' \
    --adjuster '--gpm_lookback_win 10 --gpm_noise 0.1 --max_gp_opt_steps 3000 --patience 30 --quantile 0.8 --gpm_kernel Matern32' 
done


# No HPO
etc_desc='no'

python -u run.py \
    --model TABE --model_id $test_name'_w_'$models_used'_('$etc_desc')' \
    --task_name long_term_forecast --loss 'MAE' --is_training 1 \
    \
    --data TABE --features MS --freq d --num_workers 1 \
    --target_datatype LogRet --root_path ./ --data_path dataset/btc/BTC_ret_1d.csv \
    \
    --seq_len 32 --label_len 32 --pred_len 1 --inverse \
    --batch_size 16 --lradj type3 --learning_rate 0.001 --dropout 0.1 --itr 1 --train_epochs 3  \
    --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 \
    \
    --basemodel 'SarimaModel' \
    --basemodel 'EtsModel' \
    --basemodel 'TimeMoE' \
    --basemodel 'CMamba --d_model 128 --d_ff 128 --head_dropout 0.1 --channel_mixup --gddmlp --sigma 1.0 --pscan --avg --max --reduction 2' \
    --basemodel 'iTransformer' \
    --basemodel 'DLinear' \
    --basemodel 'PatchTST' \
    --basemodel 'TimeXer' \
    --combiner '--patience 10' \
    --adjuster '--gpm_lookback_win 10 --gpm_noise 0.1 --max_gp_opt_steps 3000 --quantile 0.8 --gpm_kernel Matern32 --patience' 

