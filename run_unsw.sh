
# 2. UNSW:
nohup python -u initialize_env.py -data unsw  > logger/unsw_init.out 2>&1 &

nohup python main.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 2 -at True> logger/unsw_log_MultiLossAE2.out 2>&1 &
nohup python main.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 5 -at True> logger/unsw_log_MultiLossAE5.out 2>&1 &
nohup python main.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 10 -at True > logger/unsw_log_MultiLossAE10.out 2>&1 &
nohup python main.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -at True> logger/unsw_log_MultiLossAE20.out 2>&1 &



nohup python evaluate_attack_type.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -ep 2000 -tm 0.6 -mc 2 -at True --model_dir saved_models/MultiLossAE2/average/unsw --log_csv logs/MultiLossAE2/unsw_MultiLossAE > logger/unsw_MultiLossAE2_2000_0.6.log 2>&1 &
nohup python evaluate_attack_type.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -ep 2000 -tm 1.2 -mc 5 -at True --model_dir saved_models/MultiLossAE5/average/unsw --log_csv logs/MultiLossAE5/unsw_MultiLossAE > logger/unsw_MultiLossAE5_2000_1.1.log 2>&1 &
nohup python evaluate_attack_type.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -ep 2000 -tm 1 -mc 10 -at True --model_dir saved_models/MultiLossAE10/average/unsw --log_csv logs/MultiLossAE10/unsw_MultiLossAE > logger/unsw_MultiLossAE5_2000_1.log 2>&1 &
nohup python evaluate_attack_type.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -ep 2000 -tm 1.4 -mc 20 -at True --model_dir saved_models/MultiLossAE20/average/unsw --log_csv logs/MultiLossAE20/unsw_MultiLossAE > logger/unsw_MultiLossAE20_2000_1.4.log 2>&1 &

