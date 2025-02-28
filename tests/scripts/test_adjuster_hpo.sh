export PYTHONPATH=$PYTHONPATH:./Time-Series-Library


kernel=RBF

args_no_hpo = '--gpm_lookback_win 10 --max_gp_opt_steps 2000 --quantile 0.8' 
args_hpo1 = '--adaptive_hpo --hpo_interval 3 --max_hpo_eval 20 --max_gp_opt_steps 2000 --quantile 0.8'
args_hpo2 = '--adaptive_hpo --hpo_interval 5 --max_hpo_eval 20 --max_gp_opt_steps 2000 --quantile 0.8'
args_hpo3 = '--adaptive_hpo --hpo_interval 7 --max_hpo_eval 20 --max_gp_opt_steps 2000 --quantile 0.8'
args_hpo4 = '--adaptive_hpo --hpo_interval 9 --max_hpo_eval 20 --max_gp_opt_steps 2000 --quantile 0.8'


for adj_args in $args_no_hpo, $args_hpo1
do
python -u run.py \
    --model TABE --model_id BTC1d-Adj-hpo0-Combiner-hpo5 \
    --task_name long_term_forecast \
    --target_datatype LogRet \
    --is_training 1 \
    --data TABE --features MS --freq d --num_workers 1 \
    --root_path ./ --data_path dataset/btc/BTC_ret_1d.csv \
    --seq_len 32 --label_len 32 --pred_len 1 --inverse \
    --loss 'MAE' \
    --batch_size 16 \
    --lradj type3 --dropout 0.1 --itr 1 --train_epochs 3 --learning_rate 0.001 \
    \
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
    --adjuster $adj_args \
    --gpm_kernel $kernel 
done 

