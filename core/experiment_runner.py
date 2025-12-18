from core.data_processing import client_data_process
from core.client_factory import create_clients
from core.run_training import run_machine_learning
from agents.servers import get_server_class
from function.utils import (
    load_train_data_loader,
    load_val_data_loader,
    load_test_data_loader,
    load_mal_data_loader,
    generate_experiment_ids,
    identify_random_elements,
    filter_data_loader_by_label
)
from function.arguments import Arguments
import pandas as pd
import os
import random
from function.utils.experiment_util import create_dev_dataset_from_clients
from visualize import visualize,visualize_muitizAE
from collections import Counter
import torch

def run_exp(config):
    log_file = generate_experiment_ids(
        config["dataset"],
        config["model_type"],
        config["epochs"],
        config["poisoned_sample_ratio"],
        config["num_of_poisoned_workers"],
        config["attack_noise_std"],
        config["aggregation_type"],
        config["threshold_multiplier"],
        config["num_multi_class_clients"]
    )

    

    
    
    
    from loguru import logger
    print(log_file)
    save_dir = os.path.join(config["dataset"], config["model_type"],str(config["num_multi_class_clients"]))
    os.makedirs(save_dir, exist_ok=True)
    
    handler = logger.add(log_file, enqueue=True)
   
   
   
    args = Arguments(logger, config)
    args.log()
    

    train_data_loader_full = load_train_data_loader(logger, args)
    val_data_loader_full = load_val_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)
    mal_data_loader = load_mal_data_loader(logger, args)
    
    
    if args.model_type!="MultiZAE": #các model khác đưa hết các nhãn về 0-1
        #Du liệu train/va/ chỉ để nhãn 0-1 trước khi phân phối cho client
        # Tạo DataLoader mới với nhãn binary
        train_data_loader_full = remap_labels_binary(train_data_loader_full)
        val_data_loader_full = remap_labels_binary(val_data_loader_full)
            
    total_clients = args.num_workers
    num_multi = args.num_multi_class_clients
    num_single = total_clients - num_multi
    all_client_ids = list(range(total_clients))
    # Gán cố định: 0 đến num_single-1 là single-class
    single_class_client_ids = list(range(num_single))
    # Gán tiếp theo: num_single đến total_clients-1 là multi-class
    multi_class_client_ids = list(range(num_single, total_clients))

    args.logger.info(
                    "Multi-class client IDs: {}",
                    str(multi_class_client_ids)
                )

    args.logger.info(
                    "Single-class client IDs: {}",
                    str(single_class_client_ids)
                )

    poisoned_workers = identify_random_elements(total_clients, args.num_poisoned_workers)

    poisoned_workers_multi = [cid for cid in poisoned_workers if cid in multi_class_client_ids]
    poisoned_workers_single = [cid for cid in poisoned_workers if cid in single_class_client_ids]


    original_num_workers = args.num_workers  # Lưu lại tổng số client
    train_loaders_multi = []
    train_loaders_single = []
    if (args.noniid==False): 
        if len(multi_class_client_ids) > 0:
            # Nhóm đa lớp: sử dụng DataLoader gốc (chứa cả lớp 0 và 1 )
            args.num_workers = len(multi_class_client_ids)
            train_loaders_multi = client_data_process(
                args,
                train_data_loader_full,
                poisoned_workers_multi,
                mal_data_loader,
                args.train_batch_size,
                poison=True
            )
        
        if len(multi_class_client_ids) < original_num_workers:
            # Nhóm chỉ lớp 0: lọc DataLoader gốc để chỉ giữ lại các mẫu có nhãn 0
            train_data_loader_single = filter_data_loader_by_label(train_data_loader_full, label_to_keep=0)
            args.num_workers = len(single_class_client_ids)
            train_loaders_single = client_data_process(
                args,
                train_data_loader_single,
                poisoned_workers_single,
                mal_data_loader,
                args.train_batch_size,
                poison=True
            )
        # Gán lại cho đúng thứ tự client ban đầu bằng cách sử dụng dict
        train_loaders_dict = {}
        for i, cid in enumerate(multi_class_client_ids):
            train_loaders_dict[cid] = train_loaders_multi[i]
        for i, cid in enumerate(single_class_client_ids):
            train_loaders_dict[cid] = train_loaders_single[i]
        final_train_loaders = [train_loaders_dict[i] for i in sorted(train_loaders_dict.keys())]
        
    else:
        #Đối với dữ liệu non-iid, cả nhóm đa lớp và nhóm đơn lớp đều sử dụng DataLoader gốc
        # vì đã phân phối non-iid theo client bên trong hàm client_data_process, đồng thời sẽ có kèm điều kiện sinh nhãn chỉ mỗi nhãn o cho x clietnt đơn lớp
        args.logger.info("Processing multi-class clients non-idd' data. Total client: {}, Multiclass class: {}",(args.num_workers),(args.num_multi_class_clients))
        final_train_loaders = client_data_process(
                args,
                train_data_loader_full,
                poisoned_workers_multi,
                mal_data_loader,
                args.train_batch_size,
                poison=True
            )
        
    
    # Lấy toàn bộ nhãn trong DataLoader
    for client_id, loader in enumerate(final_train_loaders):
    
        all_labels = []
        for _, labels in loader:
            # Nếu labels là tensor, convert sang list
            all_labels += labels.cpu().tolist()
        
        # Đếm số lượng mỗi nhãn
        label_count = Counter(all_labels)
        
        # In ra log
        args.logger.info("Client {}: {}", client_id, dict(label_count))
    # Xử lý dữ liệu validation theo cách tương tự:
    val_loaders_multi = []
    val_loaders_single = []

    if len(multi_class_client_ids) > 0:
        args.num_workers = len(multi_class_client_ids)
        val_loaders_multi = client_data_process(
            args,
            val_data_loader_full,
            poisoned_workers_multi,
            mal_data_loader,
            args.val_batch_size,
            poison=True
        )

    if len(multi_class_client_ids) < original_num_workers:   
        val_data_loader_single = filter_data_loader_by_label(val_data_loader_full, label_to_keep=0)
        args.num_workers = len(single_class_client_ids)
        val_loaders_single = client_data_process(
            args,
            val_data_loader_single,
            poisoned_workers_single,
            mal_data_loader,
            args.val_batch_size,
            poison=True
        )

    val_loaders_dict = {}
    for i, cid in enumerate(multi_class_client_ids):
        val_loaders_dict[cid] = val_loaders_multi[i]
    for i, cid in enumerate(single_class_client_ids):
        val_loaders_dict[cid] = val_loaders_single[i]
    final_val_loaders = [val_loaders_dict[i] for i in sorted(val_loaders_dict.keys())]

    # Khôi phục args.num_workers về tổng số client ban đầu
    args.num_workers = original_num_workers
    
    test_data_loaders = client_data_process(
        args,
        test_data_loader,
        poisoned_workers,
        mal_data_loader,
        args.test_batch_size,
        poison=False,
    )
    # -------------------------------
    # Tạo các client từ DataLoader train, val và test
    clients = create_clients(args, final_train_loaders, final_val_loaders, test_data_loaders)

    # Chỉ tạo dev_dataset khi dùng phương pháp fedmse
    if args.aggregation_type == "FedMSE":
        args.dev_dataset = create_dev_dataset_from_clients(clients, max_samples_per_client=100)
        print("Dev_dataset created:", args.dev_dataset.shape, args.dev_dataset.dtype)


    # # Thêm vào đây để kiểm tra labels từng client:
    # count_labels_per_client(args, clients)

    args.set_train_log_df(
        pd.DataFrame(
            columns=["epoch", "client_id", "is_mal", "train_re", "train_latent_z", "train_loss",
                     "threshold_re", "threshold_z", "best_val_loss", "best_epoch", "is_training"]
        )
    )
    args.set_test_log_df(
        pd.DataFrame(
            columns=["epoch", "client_id", "is_mal", "auc", "accuracy", "precision", "recall", "f1"]
        )
    )

    ServerClass = get_server_class(args.model_type)
    server = ServerClass(args)

    epoch_stop=run_machine_learning(server, clients, poisoned_workers, args)
    
    save_dir = os.path.join(args.dataset, args.model_type, str(args.num_multi_class_clients))
    os.makedirs(save_dir, exist_ok=True)
    
    log_name = log_file.split(".log")[0]
    
    train_csv_file = f"{log_name}_train.csv"
    test_csv_file  = f"{log_name}_test.csv"
    
    args.get_train_log_df().to_csv(train_csv_file)
    args.get_test_log_df().to_csv(test_csv_file)
    if args.model_type=="MultiZAE":
        visualize_muitizAE(train_csv_file,log_name,1)

    else:
        visualize(train_csv_file,log_name,1)

    logger.remove(handler)

    server.test_on_clients(epoch_stop, clients, poisoned_workers)



def count_labels_per_client(args, clients):
    def count_labels(data_loader):
        label_counter = Counter()
        for _, labels in data_loader:
            label_counter.update(labels.cpu().numpy().tolist())
        return label_counter

    for i, client in enumerate(clients):
        args.logger.info(f"\n📦 Client {i}:")

        train_labels = count_labels(client.train_data_loader)
        val_labels = count_labels(client.val_data_loader)
        test_labels = count_labels(client.test_data_loader)

        args.logger.info("  📊 Train Label Counts: {}", dict(train_labels))
        args.logger.info("  🧪 Val Label Counts:   {}", dict(val_labels))
        args.logger.info("  🧪 Test Label Counts:  {}", dict(test_labels))

def remap_labels_binary(data_loader):
    """
    Trả về DataLoader mới với nhãn: 0 giữ nguyên, nhãn >=1 → 1.
    """
    X_list = []
    Y_list = []

    # thu dữ liệu từ DataLoader gốc
    for x, y in data_loader:
        X_list.append(x)
        Y_list.append(y)

    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)

    # Ánh xạ: 0 → 0, nhãn ≥1 → 1
    Y = (Y >= 1).long()   # chuyển thành 1 nếu y >=1

    # Tạo DataLoader mới
    dataset = torch.utils.data.TensorDataset(X, Y)
    new_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_loader.batch_size,
        shuffle=data_loader.shuffle if hasattr(data_loader, "shuffle") else False
    )
    return new_loader