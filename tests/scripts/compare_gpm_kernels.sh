export PYTHONPATH=$PYTHONPATH:./Time-Series-Library

test_name='Adj_gpm_knls'
desc='all'

for kernel in RBF Matern32 Matern52 Linear Brownian ; do
python -u run.py \
    --model TABE \
    --model_id $test_name'_('$desc')_'$kernel \
    --data TABE_FILE --data_path 'BTC-USD_LogRet_2021-01-01_2023-01-01_1d.csv' \
    --train_epochs 3  \
    --adjuster '--gpm_kernel '$kernel \
    --basemodel 'EtsModel' \
    --basemodel 'SarimaModel' \
    --basemodel 'DLinear --batch_size 8' \
    --basemodel 'iTransformer' \
    --basemodel 'PatchTST --batch_size 16' \
    --basemodel 'TimeXer' \
    --basemodel 'CMamba --batch_size 64 --lradj type3 --learning_rate 0.0005 --d_model 128 --d_ff 128 ' \
    --basemodel 'TimeMoE' 
done

