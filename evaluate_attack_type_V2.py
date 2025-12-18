import os
import torch
import argparse
import pandas as pd
from loguru import logger
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import glob
from collections import defaultdict

from function.arguments import Arguments
from agents.clients import get_client_class
from core.data_processing import client_data_process
from core.client_factory import create_clients
from function.utils import load_test_data_loader
import ast

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved models with thresholds from log")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name (e.g., cic_ids, nslkdd)")
    parser.add_argument("-m", "--model_type", required=True, help="Model type (AE, DAE, SAE, SDAE, SupAE, DualLossAE, PTL)")
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=128, help="Train batch size")
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=128, help="Val batch size")
    parser.add_argument("-di", "--dimension", type=int, default=128, help="Dimension")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-ep", "--epochs", type=int, default=4000, help="Epochs")
    parser.add_argument("-agg", "--aggregation_type", type=str, default="average", help="Aggregation type")
    parser.add_argument("-nt", "--noise_type", type=str, default="label_flipping", help="Noise type")
    parser.add_argument("-pw", "--poisoned_workers", type=int, default=0, help="Number of poisoned clients")
    parser.add_argument("-pr", "--poisoned_ratio", type=float, default=1.0, help="Poisoned sample ratio")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.001, help="Noise stddev")
    parser.add_argument("-ans", "--attack_noise_std", type=float, default=3.0, help="Attack noise stddev")
    parser.add_argument("-cs", "--coef_shrink_ae", type=float, default=1.0, help="Coefficient shrink AE")
    parser.add_argument("-tm", "--threshold_multiplier", type=float, default=0.0, help="Threshold multiplier")
    parser.add_argument("-mc", "--num_multi_class_clients", type=int, default=0, help="Multi-class client count")
    parser.add_argument("-at", "--by_attack_type", type=bool, default=False, help="By attack type (False/ True)")
    parser.add_argument("-noniid", "--noniid", type=bool, default=False, help="NonIID (False/ True)")
    parser.add_argument("--model_dir", required=True, help="Directory containing saved models")
    parser.add_argument("--log_csv", required=True, help="Prefix of CSV log file (e.g., logs/unsw_DualLossAE_)")
    parser.add_argument("--output_dir", default="logs", help="Directory to save results")
    return parser.parse_args()

def build_config(args_ns):
    return {
        "dataset": args_ns.dataset,
        "train_batch_size": args_ns.train_batch_size,
        "val_batch_size": args_ns.val_batch_size,
        "test_batch_size": 1,
        "mal_batch_size": args_ns.val_batch_size,
        "dimension": args_ns.dimension,
        "epochs": args_ns.epochs,
        "model_type": args_ns.model_type,
        "noise_type": args_ns.noise_type,
        "num_of_poisoned_workers": args_ns.poisoned_workers,
        "poisoned_sample_ratio": args_ns.poisoned_ratio,
        "learning_rate": args_ns.learning_rate,
        "noise_std": args_ns.noise_std,
        "attack_noise_std": args_ns.attack_noise_std,
        "aggregation_type": args_ns.aggregation_type,
        "coef_shrink_ae": args_ns.coef_shrink_ae,
        "threshold_multiplier": args_ns.threshold_multiplier,
        "num_multi_class_clients": args_ns.num_multi_class_clients,
        "by_attack_type": args_ns.by_attack_type,
        "noniid": args_ns.noniid,
    }

def find_log_file(log_prefix: str) -> str:
    matched_files = glob.glob(f"{log_prefix}*train.csv")
    if not matched_files:
        raise FileNotFoundError(f"No log CSV file found with prefix: {log_prefix} and suffix '*train.csv'")
    if len(matched_files) > 1:
        raise ValueError(f"Multiple log files found with prefix '{log_prefix}': {matched_files}")
    return matched_files[0]

def find_latest_valid_epoch(log_df, client_id, target_epoch):
    valid_rows = log_df[(log_df["client_id"] == client_id) & (log_df["epoch"] <= target_epoch)]
    valid_rows = valid_rows[(valid_rows["epoch"] - 1) % 10 == 0]
    if valid_rows.empty:
        return None
    return valid_rows["epoch"].max() - 1

def main():
    args_ns = parse_args()
    config = build_config(args_ns)
    args = Arguments(logger, config)
    args.by_attack_type = True

    os.makedirs(args_ns.output_dir, exist_ok=True)

    log_csv_path = find_log_file(args_ns.log_csv)
    df_log = pd.read_csv(log_csv_path)

    test_data_loader = load_test_data_loader(logger, args)
    test_data_loaders = client_data_process(
        args, test_data_loader, None, None, args.test_batch_size, poison=False
    )
    clients = create_clients(args, [None]*args.num_workers, [None]*args.num_workers, test_data_loaders)

    logger.info("🔍 Evaluate ACC per attack type for model: {}, epoch: {}", args.model_type, args_ns.epochs)

    summary_by_type = defaultdict(list) 
    all_client_results = []

    for client_idx, client in enumerate(clients):
        fallback_epoch = args_ns.epochs
        row = df_log[(df_log["epoch"] == args_ns.epochs) & (df_log["client_id"] == client_idx)]
        if row.empty:
            fallback_epoch = find_latest_valid_epoch(df_log, client_idx, args_ns.epochs - 1)
            if fallback_epoch is None:
                logger.warning(f"⚠️ No valid log info for client {client_idx + 1} up to epoch {args_ns.epochs}")
                continue
            logger.warning(f"↩️ Falling back to epoch {fallback_epoch} for client {client_idx + 1}")
            row = df_log[(df_log["epoch"] == fallback_epoch) & (df_log["client_id"] == client_idx)]
        row = row.iloc[0]
        threshold_re = eval(row["threshold_re"]) if pd.notna(row["threshold_re"]) else (0, 0)
        threshold_z = eval(row["threshold_z"]) if pd.notna(row["threshold_z"]) else (0, 0)
        
        if(args.model_type=="MultiZAE"):
            best_z_target = (eval(row["best_z_target"]) if pd.notna(row["best_z_target"]) else None)
            if best_z_target is not None:
               # Nếu là string thì mới cần literal_eval
                if isinstance(best_z_target, str):
                    z_list = ast.literal_eval(best_z_target)
                else:
                    z_list = best_z_target   # đã là list sẵn rồi

                Z_target = torch.tensor(z_list, dtype=torch.float32)
            else:
                Z_target = None
            client.set_best_ckpt(0, fallback_epoch, threshold_re, threshold_z, None,Z_target)
        else:
            client.set_best_ckpt(0, fallback_epoch, threshold_re, threshold_z, None)
            
        model_path = os.path.join(args_ns.model_dir, f"epoch_{fallback_epoch}_client_{client_idx}.pt")
        client.update_nn_parameters(torch.load(model_path, map_location=client.device))

        if args.model_type=="MultiZAE":
            results = client.test_two_side_by_attack_type(
                Z_target,
                verbose=False
            )
        else:
            results = client.test_by_attack_type_full(
                threshold_re[0] + args.threshold_multiplier * threshold_re[1],
                threshold_z[0] + args.threshold_multiplier * threshold_z[1],
                verbose=False
            )

        logger.info(f"\n📊 Client {client_idx + 1} - Metrics by Attack Type")
        logger.info(f"{'Attack Type':<15} {'ACC':>8} {'PRE':>8} {'REC':>8} {'F1':>8} {'Support':>8}")

        for atk_type, metric in sorted(results.items()):
            acc = metric["acc"]
            prec = metric["precision"]
            recall = metric["recall"]
            f1 = metric["f1-score"]
            support = metric["support"]

            logger.info(
                f"{atk_type:<15} "
                f"{acc:>8.4f} "
                f"{prec:>8.4f} "
                f"{recall:>8.4f} "
                f"{f1:>8.4f} "
                f"{support:>8}"
            )

            summary_by_type[atk_type].append((acc, prec, recall, f1, support))

            # Lưu kết quả từng client
            all_client_results.append({
                "dataset": args_ns.dataset,
                "model": args_ns.model_type,
                "epoch": fallback_epoch,
                "client_id": client_idx + 1,
                "attack_type": atk_type,
                "acc": acc,
                "precision": prec,
                "recall": recall,
                "f1": f1,
                "support": support
            })

    # Lưu client-level CSV
    client_csv_path = f"logs/{args_ns.dataset}/{args_ns.model_type}/{args_ns.num_multi_class_clients}/{args_ns.dataset}_{args_ns.model_type}_{args_ns.epochs}_{args_ns.num_multi_class_clients}_clients.csv"
    pd.DataFrame(all_client_results).to_csv(client_csv_path, index=False)
    logger.info(f"✅ Saved client-level results to: {client_csv_path}")

    # Tổng hợp weighted
    logger.info(
        f"{'Attack Type':<15} "
        f"{'W-ACC':>10} {'W-PRE':>10} {'W-REC':>10} {'W-F1':>10} "
        f"{'Samples':>10} {'Clients':>10}"
    )

    overall_acc_sum = overall_pre_sum = overall_rec_sum = overall_f1_sum = overall_total_support = 0
    summary_rows = []

    for atk_type in sorted(summary_by_type.keys()):
        accs = [x[0] for x in summary_by_type[atk_type]]
        pres = [x[1] for x in summary_by_type[atk_type]]
        recs = [x[2] for x in summary_by_type[atk_type]]
        f1s  = [x[3] for x in summary_by_type[atk_type]]
        supports = [x[4] for x in summary_by_type[atk_type]]

        total_support = sum(supports)
        num_clients = len(accs)

        if total_support > 0:
            w_acc = np.average(accs, weights=supports)
            w_pre = np.average(pres, weights=supports)
            w_rec = np.average(recs, weights=supports)
            w_f1  = np.average(f1s,  weights=supports)
        else:
            w_acc = w_pre = w_rec = w_f1 = 0.0

        overall_acc_sum += w_acc * total_support
        overall_pre_sum += w_pre * total_support
        overall_rec_sum += w_rec * total_support
        overall_f1_sum  += w_f1  * total_support
        overall_total_support += total_support

        summary_rows.append({
            "attack_type": atk_type,
            "weighted_acc": w_acc,
            "weighted_precision": w_pre,
            "weighted_recall": w_rec,
            "weighted_f1": w_f1,
            "total_samples": total_support,
            "num_clients": num_clients
        })

        logger.info(
            f"{atk_type:<15} "
            f"{w_acc:>10.4f} {w_pre:>10.4f} {w_rec:>10.4f} {w_f1:>10.4f} "
            f"{total_support:>10} {num_clients:>10}"
        )

    # Lưu summary CSV
    summary_csv_path = f"logs/{args_ns.dataset}/{args_ns.model_type}/{args_ns.num_multi_class_clients}/{args_ns.dataset}_{args_ns.model_type}_{args_ns.epochs}_{args_ns.num_multi_class_clients}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    logger.info(f"✅ Saved summary results to: {summary_csv_path}")

    # Lưu overall TXT
    overall_path = f"logs/{args_ns.dataset}/{args_ns.model_type}/{args_ns.num_multi_class_clients}/{args_ns.dataset}_{args_ns.model_type}_{args_ns.epochs}_{args_ns.num_multi_class_clients}_overall.txt"
    with open(overall_path, "w", encoding="utf-8") as f:
        if overall_total_support > 0:
            overall_acc = overall_acc_sum / overall_total_support
            overall_pre = overall_pre_sum / overall_total_support
            overall_rec = overall_rec_sum / overall_total_support
            overall_f1  = overall_f1_sum  / overall_total_support

            f.write(f"Overall ACC: {overall_acc:.6f}\n")
            f.write(f"Overall Precision: {overall_pre:.6f}\n")
            f.write(f"Overall Recall: {overall_rec:.6f}\n")
            f.write(f"Overall F1: {overall_f1:.6f}\n")

            logger.info("\n📌 Overall metrics saved:")
            logger.info(f"Overall ACC: {overall_acc:.4f}")
            logger.info(f"Overall Precision: {overall_pre:.4f}")
            logger.info(f"Overall Recall: {overall_rec:.4f}")
            logger.info(f"Overall F1: {overall_f1:.4f}")
        else:
            f.write("⚠️ No support data available.\n")
            logger.warning("⚠️ No support data available. Cannot compute overall metrics.")

if __name__ == "__main__":
    main()
