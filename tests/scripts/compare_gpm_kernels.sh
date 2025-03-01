export PYTHONPATH=$PYTHONPATH:./Time-Series-Library

test_name='Cmpr_gpm_knls'
models_used='all'
etc_desc='C_ahpo_A_w10'

for kernel in RBF Matern32 Matern52 Linear Brownian
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
    --basemodel 'CMamba --d_model 128 --d_ff 128 --head_dropout 0.1 --channel_mixup --gddmlp --sigma 1.0 --pscan --avg --max --reduction 2' \
    --basemodel 'iTransformer' \
    --basemodel 'DLinear' \
    --basemodel 'PatchTST' \
    --basemodel 'TimeXer' \
    --basemodel 'TimeMoE' \
    --combiner '--adaptive_hpo --hpo_interval 5 --max_hpo_eval 100' \
    --adjuster '--gpm_lookback_win 10 --max_gp_opt_steps 2000 --quantile 0.8 --gpm_kernel '$kernel 
done

