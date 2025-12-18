@echo off

REM === 1. Không dùng START (để nó hiện trực tiếp ra terminal) ===

powershell -Command "python main.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 5  -at True 2>&1 | Tee-Object -FilePath logger/unsw_log_MultiLossAE5.out"

powershell -Command "python main.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 10 -at True 2>&1 | Tee-Object -FilePath logger/unsw_log_MultiLossAE10.out"

powershell -Command "python main.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 2000 -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 1 -cs 1 -tm 1 -mc 20 -at True 2>&1 | Tee-Object -FilePath logger/unsw_log_MultiLossAE20.out"
