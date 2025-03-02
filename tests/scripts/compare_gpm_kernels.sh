export PYTHONPATH=$PYTHONPATH:./Time-Series-Library

test_name='Adj_gpm_knls'
models_used='all'
etc_desc=''

for kernel in RBF Matern32 Matern52 Linear Brownian
do
python -u run.py \
    --model TABE --model_id $test_name'_w_'$models_used'_('$etc_desc')' \
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
    --adjuster '--patience 30 --gpm_kernel '$kernel 
done

