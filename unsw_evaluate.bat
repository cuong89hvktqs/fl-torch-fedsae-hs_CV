REM ============================
REM 3. EVALUATION SCRIPTS
REM ============================

@echo off

@echo off

REM === Chạy và lưu log, hiển thị trực tiếp trong Anaconda Prompt ===

powershell -Command "python evaluate_attack_type.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -ep 2000 -tm 0.6 -mc 2 -at True --model_dir saved_models/MultiLossAE/2/average/unsw --log_csv logs/MultiLossAE/2/unsw_MultiLossAE 2>&1 | Tee-Object -FilePath logger/unsw_MultiLossAE2_2000_0.6.log"

powershell -Command "python evaluate_attack_type.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -ep 2000 -tm 1.2 -mc 5 -at True --model_dir saved_models/MultiLossAE/5/average/unsw --log_csv logs/MultiLossAE/5/unsw_MultiLossAE 2>&1 | Tee-Object -FilePath logger/unsw_MultiLossAE5_2000_1.2.log"

powershell -Command "python evaluate_attack_type.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -ep 2000 -tm 1 -mc 10 -at True --model_dir saved_models/MultiLossAE/10/average/unsw --log_csv logs/MultiLossAE/10/unsw_MultiLossAE 2>&1 | Tee-Object -FilePath logger/unsw_MultiLossAE10_2000_1.log"

powershell -Command "python evaluate_attack_type.py -d unsw -m MultiLossAE -tbs 128 -vbs 1 -di 196 -ep 2000 -tm 1.4 -mc 20 -at True --model_dir saved_models/MultiLossAE/20/average/unsw --log_csv logs/MultiLossAE/20/unsw_MultiLossAE 2>&1 | Tee-Object -FilePath logger/unsw_MultiLossAE20_2000_1.4.log"
