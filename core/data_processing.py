from function.utils import (
    distribute_batches_equally,
    distribute_non_iid_dirichlet,
    convert_distributed_data_into_numpy,
    poison_data,
    generate_data_loaders_from_distributed_dataset,
    filter_data_loader_by_label,distribute_non_iid_dirichlet_zero_lable_client
)

def client_data_process(args, data_loader, poisoned_workers, mal_data_loader, batch_size, poison=True):
    if args.noniid:
        args.logger.info("Distributing data in Non-IID Dirichlet manner, num_workers: {}, num_multi_class_clients: {}".format(args.num_workers, args.num_multi_class_clients))
        if(args.num_workers > args.num_multi_class_clients):
            distributed_dataset = distribute_non_iid_dirichlet_zero_lable_client(
                data_loader,
                args.num_workers,
                alpha=0.4,
                num_zero_label_clients= args.num_workers - args.num_multi_class_clients,# số clietnt chỉ có nhãn 0
                zero_label=0 # nhãn 0
            )
        else:
            distributed_dataset = distribute_non_iid_dirichlet(
                data_loader,
                args.num_workers,
                alpha=0.5,
            )
    else:
        distributed_dataset = distribute_batches_equally(data_loader, args.num_workers)
    
    distributed_dataset = convert_distributed_data_into_numpy(distributed_dataset)
    
    if poison:
        distributed_dataset = poison_data(
            args.logger,
            distributed_dataset,
            args.num_workers,
            poisoned_workers,
            args.noise_type,
            mal_data_loader,
            args.replacement_ratio,
            args.attack_std_noise,
        )

    data_loaders = generate_data_loaders_from_distributed_dataset(distributed_dataset, batch_size)
    return data_loaders
