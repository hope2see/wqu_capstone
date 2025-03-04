export PYTHONPATH=$PYTHONPATH:./Time-Series-Library


test_name='Adj_HP_Grid'
desc='chosen_models_BTC'


for gpm_lookback_win in 10 30 50 -1; do 
for adj_eval_win in 3 5 10; do 
for adj_cred_factor in 5 10 50 100; do 
python -u run.py \
    --model TABE \
    --model_id $test_name'_('$desc')_'$gpm_lookback_win'_'$adj_eval_win'_'$adj_cred_factor \
    --data TABE_FILE --data_path 'BTC-USD_LogRet_2021-01-01_2023-01-01_1d.csv' \
    --train_epochs 3  \
    --adjuster '--patience 30 --gpm_lookback_win '$gpm_lookback_win' --adj_eval_win  '$adj_eval_win' --adj_cred_factor '$adj_cred_factor \
    --basemodel 'EtsModel' \
    --basemodel 'SarimaModel' \
    --basemodel 'TimeMoE' \
    --basemodel 'Drifter' \
    --basemodel 'Noiser'
done 
done
done 
