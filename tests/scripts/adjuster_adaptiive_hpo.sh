export PYTHONPATH=$PYTHONPATH:./Time-Series-Library

test_name='Adj_ahpo'
models_used='S_E_TM'


# Adaptive HPO
etc_desc=''

# HPO
for interval in 3 6 10 20 50 1000; do
python -u run.py \
    --model TABE --model_id $test_name'_w_'$models_used'_('$etc_desc'_'$interval')' \
    --task_name long_term_forecast --loss 'MAE' --is_training 1 \
    \
    --data TABE_ONLINE --features MS --freq d --num_workers 1 \
    \
    --seq_len 32 --label_len 32 --pred_len 1 --inverse \
    --batch_size 16 --lradj type3 --learning_rate 0.001 --dropout 0.1 --itr 1 --train_epochs 3  \
    --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 \
    \
    --basemodel 'SarimaModel' \
    --basemodel 'EtsModel' \
    --basemodel 'TimeMoE' \
    --adjuster '--adaptive_hpo --hpo_interval '$interval' --max_hpo_eval 20 --patience 30'
    # --basemodel 'CMamba --d_model 128 --d_ff 128 --head_dropout 0.1 --channel_mixup --gddmlp --sigma 1.0 --pscan --avg --max --reduction 2' \
    # --basemodel 'iTransformer' \
    # --basemodel 'DLinear' \
    # --basemodel 'PatchTST' \
    # --basemodel 'TimeXer' \
done
