import torch
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import os
import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict


class ClientMultiZAE:
    def __init__(
        self, args, client_idx, train_data_loader, val_data_loader, test_data_loader
    ):
        """
        :param args: experiment arguments
        :param client_idx: Client index
        :param train_data_loader: Training data loader
        :param val_data_loader: Val data loader
        :param test_data_loader: Test data loader
        """
        self.args = args
        self.client_idx = client_idx
        self.model_type = self.args.model_type
        self.is_training = True

        self.best_loss = 1e9
        self.best_epoch = -1
        self.threshold_re = 1e9
        self.threshold_z = 1e9
        self.z_target = None
        self.best_weight_model = None

        # use for log, not for train/test
        self.recent_re = 0.0
        self.recent_latent_z = 0.0
        self.recent_train_loss = 0.0
        self.recent_val_loss = 0.0
        self.recent_threshold_re = (1e9, 0)
        self.recent_threshold_z = (1e9, 0)
        self.recent_z_target = None
        
        self.device = self.initialize_device()
        self.set_net(self.load_default_model())

        self.loss_function = self.args.loss_function()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.learning_rate)

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        
         

    def initialize_device(self):
        """
        Creates appropriate torch device for client operation.
        """
        if torch.cuda.is_available() and self.args.cuda:
            # print("Device: GPU")
            return torch.device("cuda:0")
        else:
            # print("Device: CPU")
            return torch.device("cpu")

    def set_net(self, net):
        """
        Set the client's NN.
        """
        self.net = net
        self.net.to(self.device)

    def set_best_ckpt(
        self, best_loss, best_epoch, threshold_re, threshold_z, best_weight_model, best_z_target
    ):
        """
        Set the best training point.
        """
        self.best_loss = best_loss
        self.best_epoch = best_epoch
        self.threshold_re = threshold_re
        self.threshold_z = threshold_z
        self.best_weight_model = copy.deepcopy(best_weight_model)
        self.z_target = best_z_target

    def set_training_status(self, training_status):
        """
        Set the training status.
        """
        self.is_training = training_status

    def set_recent_metric(
        self,
        recent_re,
        recent_latent_z,
        recent_train_loss,
        recent_val_loss,
        recent_threshold_re,
        recent_threshold_z,
        recent_z_target,
    ):
        """
        Set the recent metrics for logging.
        """
        self.recent_re = recent_re
        self.recent_latent_z = recent_latent_z
        self.recent_train_loss = recent_train_loss
        self.recent_val_loss = recent_val_loss
        self.recent_threshold_re = recent_threshold_re
        self.recent_threshold_z = recent_threshold_z
        self.recent_z_target = recent_z_target

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = self.args.get_net(self.model_type)
        default_model_path = os.path.join(
            self.args.default_model_folder_path, model_class.__name__ + ".model"
        )

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.
        """
        print("Model type: ",self.model_type)
        model_class = self.args.get_net(self.model_type)
        model = model_class(self.args.dimension)

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.args.logger.warning(
                    "Couldn't load model. Attempting to map CUDA tensors to CPU to solve error."
                )

                model.load_state_dict(
                    torch.load(model_file_path, map_location=torch.device("cpu"))
                )
        else:
            self.args.logger.warning("Could not find model: {}".format(model_file_path))

        return model

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()

    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def calculate_loss(self, input, label,latent_loss_type="mse"):
        """
        Calculate the loss value.

        :param input: Training data sample
        """
        return self.net.calculate_loss(input, label,latent_loss_type)

    def train(self, epoch, pdf_writer_z):
        self.net.train()
        for input, label in self.train_data_loader:
            input, label = input.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            total_loss, re_loss,latent_loss = self.calculate_loss(input, label,latent_loss_type="mse")
            total_loss.backward()
            self.optimizer.step()
        return total_loss, re_loss,latent_loss

    #hàm này đang tính phân biệt bất thường theo 2 cách:
    # C1: threshold_z theo nguongx ép về 0 cứng nhắc
    #C2: Lưu Z_target mềm mại hơn dựa trên prototype từ server để phân loại
    def validate(self, epoch, k_sigma=3, latent_loss_type="mse"):
        self.net.eval()

        y_true = []
        scores_one_side = []
        scores_two_side = []
        re_loss_label_0 = []
        z_norm_list = []

        total_val_loss = 0.0
        re_val_loss = 0.0
        latent_val_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            Z_target = self.net.weighted_attack_target(self.device)

            for input, label in self.val_data_loader:
                input, label = input.to(self.device), label.to(self.device)

                z, decode = self.net(input)

                total_loss, re_loss, latent_loss = self.calculate_loss(
                    input, label, latent_loss_type
                )

                total_val_loss += total_loss.item()
                re_val_loss += re_loss.item()
                latent_val_loss += latent_loss.item()
                n_batches += 1

                # ===== RE per sample =====
                re_sample = ((decode - input) ** 2).mean(dim=1)

                mask_normal = (label == 0)
                if mask_normal.any():
                    re_loss_label_0.extend(re_sample[mask_normal].cpu().numpy())

                # ===== ONE SIDE SCORE =====
                score_1 = (z ** 2).sum(dim=1)

                # ===== TWO SIDE SCORE =====
                d0 = (z ** 2).sum(dim=1)
                d1 = ((z - Z_target) ** 2).sum(dim=1)
                score_2 = d0 - d1

                if mask_normal.any():
                    z_norm_list.extend(score_1[mask_normal].cpu().numpy())

                y_bin = (label != 0).int()
                y_true.extend(y_bin.cpu().numpy())

                scores_one_side.extend(score_1.cpu().numpy())
                scores_two_side.extend(score_2.cpu().numpy())

        # ===== THRESHOLD RE (mean, std) =====
        threshold_re = (
            (np.mean(re_loss_label_0), np.std(re_loss_label_0))
            if len(re_loss_label_0) > 0
            else (0.0, 0.0)
        )

        # ===== THRESHOLD Z (mean, std) =====
        threshold_z = (
            (np.mean(z_norm_list), np.std(z_norm_list))
            if len(z_norm_list) > 0
            else (0.0, 0.0)
        )

        total_val_loss /= n_batches
        re_val_loss /= n_batches
        latent_val_loss /= n_batches

        return {
            "threshold_re": threshold_re,
            "threshold_z": threshold_z,
            "Z_target": Z_target.detach().cpu(),
            "total_val_loss": total_val_loss,
            "re_val_loss": re_val_loss,
            "latent_val_loss": latent_val_loss,
        }


    
    
    
    
    def test(self, int_test_type=2):# test chỉ đưa về có 2 nhãn
        k_sigma_list = np.arange(3, 3.1, 0.2)
        acc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        roc_list = []
        
        if int_test_type == 1:
            for k in k_sigma_list:
                acc, pre, rec, f1, auc = self.test_one_side_fixed_threshold(self.threshold_z,k)
                acc_list.append(acc)
                precision_list.append(pre)
                recall_list.append(rec)
                f1_list.append(f1)
                roc_list.append(auc) 
        elif int_test_type == 2:
            acc, pre, rec, f1, auc = self.test_two_side(self.z_target)
            acc_list.append(acc)
            precision_list.append(pre)
            recall_list.append(rec)
            f1_list.append(f1)
            roc_list.append(auc)
        else:
            for k in k_sigma_list:
                acc, pre, rec, f1, auc = self.test_re_fixed_threshold(self.threshold_re,k)
                acc_list.append(acc)
                precision_list.append(pre)
                recall_list.append(rec)
                f1_list.append(f1)
                roc_list.append(auc)
            
        return acc_list, precision_list, recall_list, f1_list, roc_list    
            
    
    
    def test_re_fixed_threshold(self, threshold_re, k_sigma=3):# test chỉ đưa về có 2 nhãn
        self.net.eval()

        mu, sigma = threshold_re
        threshold = mu + k_sigma * sigma

        y_true, y_pred, scores = [], [], []

        with torch.no_grad():
            for input, label in self.test_data_loader:
                input, label = input.to(self.device), label.to(self.device)

                _, decode = self.net(input)
                re_loss = ((decode - input) ** 2).mean(dim=1)

                score = re_loss.cpu().numpy()
                pred = (score > threshold).astype(int)
                y_bin = (label != 0).int().cpu().numpy()

                y_true.extend(y_bin)
                y_pred.extend(pred)
                scores.extend(score)

        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
         # Tính ROC AUC (an toàn)
        if len(np.unique(y_true)) < 2:
            auc =None
        else:
            auc = roc_auc_score(y_true, scores)

        return acc, pre, rec, f1, auc

    
    def test_one_side_fixed_threshold(self, threshold_z, k_sigma=3):# test chỉ đưa về có 2 nhãn
        self.net.eval()

        mu, sigma = threshold_z
        threshold = mu + k_sigma * sigma

        y_true, y_pred, scores = [], [], []

        with torch.no_grad():
            for input, label in self.test_data_loader:
                input, label = input.to(self.device), label.to(self.device)

                z, _ = self.net(input)
                score = (z ** 2).sum(dim=1)

                pred = (score > threshold).int()
                y_bin = (label != 0).int()

                y_true.extend(y_bin.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                scores.extend(score.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
         # Tính ROC AUC (an toàn)
        if len(np.unique(y_true)) < 2:
            auc =None
        else:
            auc = roc_auc_score(y_true, scores)

        return acc, pre, rec, f1, auc
    
    def test_two_side(self, Z_target): # test chỉ đưa về có 2 nhãn
        """
        Phân loại 2 phía:
        d0 = ||z||^2
        d1 = ||z - Z_target||^2

        d0 - d1 > 0  -> attack (1)
        d0 - d1 <= 0 -> normal (0)
        """

        self.net.eval()

        y_true = []
        y_pred = []
        scores = []

        Z_target = Z_target.to(self.device)

        with torch.no_grad():
            for input, label in self.test_data_loader:
                input, label = input.to(self.device), label.to(self.device)

                z, _ = self.net(input)

                d0 = (z ** 2).sum(dim=1)
                d1 = ((z - Z_target) ** 2).sum(dim=1)

                score = d0 - d1

                pred = (score > 0).int()

                y_bin = (label != 0).int()

                y_true.extend(y_bin.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                scores.extend(score.cpu().numpy())

        # ===== METRICS =====
        acc = accuracy_score(y_true, y_pred)
        
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
         # Tính ROC AUC (an toàn)
        if len(np.unique(y_true)) < 2:
            auc =None
        else:
            auc = roc_auc_score(y_true, scores)

        return acc, pre, rec, f1, auc
     
    def test_two_side_by_attack_type(self, Z_target, verbose=False):
        """
        Evaluate two-side detection but compute metrics per attack type (1..K).
        Normal (0) is excluded from per-type evaluation.
        """

        from collections import defaultdict
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

        self.net.eval()
        Z_target = Z_target.to(self.device)

        raw_labels = []     # original label: 0..9
        bin_labels = []     # binary: 0=normal, 1=attack
        preds = []          # prediction: 0/1

        with torch.no_grad():
            for input, label in self.test_data_loader:
                input = input.to(self.device)
                label = label.to(self.device)

                z, _ = self.net(input)

                d0 = (z ** 2).sum(dim=1)
                d1 = ((z - Z_target) ** 2).sum(dim=1)
                score = d0 - d1

                pred = (score > 0).int()
                y_bin = (label != 0).int()

                raw_labels.extend(label.cpu().tolist())
                bin_labels.extend(y_bin.cpu().tolist())
                preds.extend(pred.cpu().tolist())

        # GROUP RESULTS BY ATTACK TYPE
        results_by_type = defaultdict(lambda: {"y_true": [], "y_pred": []})

        for y_true, y_pred, atk_type in zip(bin_labels, preds, raw_labels):

            # skip normal class in per-attack evaluation
            if atk_type == 0:
                continue

            results_by_type[int(atk_type)]["y_true"].append(int(y_true))
            results_by_type[int(atk_type)]["y_pred"].append(int(y_pred))

        metrics_by_type = {}

        for atk_type, data in results_by_type.items():
            y_t = data["y_true"]
            y_p = data["y_pred"]

            report = classification_report(
                y_t, y_p, output_dict=True, zero_division=0
            )
            acc = accuracy_score(y_t, y_p)

            metrics_by_type[atk_type] = {
                "acc": acc,
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"],
                "f1-score": report["1"]["f1-score"],
                "support": len(y_t),
            }

            if verbose:
                print(f"\n=== Attack Type {atk_type} ===")
                print(f"Accuracy: {acc:.4f}")
                print(classification_report(y_t, y_p, zero_division=0))
                print("Confusion Matrix:")
                print(confusion_matrix(y_t, y_p))

        return metrics_by_type


        
    def test_two_side_by_attack_type_old(self, Z_target, verbose=False): #test có nhiều nhãn
        """
        Phân loại two-side nhưng gom kết quả theo từng attack type (0..9).
        Giống như test_by_attack_type_full.
        """

        self.net.eval()

        Z_target = Z_target.to(self.device)

        # Danh sách lưu kết quả
        raw_labels = []     # label gốc (0..9)
        bin_labels = []     # 0 = normal, 1 = attack
        preds = []          # dự đoán 0/1
        scores = []         # score = d0 - d1

        with torch.no_grad():
            for input, label in self.test_data_loader:
                input = input.to(self.device)
                label = label.to(self.device)

                z, _ = self.net(input)

                # Two-side distance
                d0 = (z ** 2).sum(dim=1)
                d1 = ((z - Z_target) ** 2).sum(dim=1)
                score = d0 - d1

                # Dự đoán: score > 0 -> attack
                pred = (score > 0).int()

                # Nhãn nhị phân của true label
                y_bin = (label != 0).int()

                # Lưu kết quả
                raw_labels.extend(label.cpu().tolist())
                bin_labels.extend(y_bin.cpu().tolist())
                preds.extend(pred.cpu().tolist())
                scores.extend(score.cpu().tolist())

        # === GOM KẾT QUẢ THEO TỪNG ATTACK TYPE ===
        results_by_type = defaultdict(lambda: {"y_true": [], "y_pred": []})

        for y_true, y_pred, atk_type in zip(bin_labels, preds, raw_labels):
            results_by_type[int(atk_type)]["y_true"].append(int(y_true))
            results_by_type[int(atk_type)]["y_pred"].append(int(y_pred))

        # === TÍNH METRICS ===
        metrics_by_type = {}

        for atk_type, data in results_by_type.items():
            y_t = data["y_true"]
            y_p = data["y_pred"]

            report = classification_report(
                y_t, y_p, output_dict=True, zero_division=0
            )
            acc = accuracy_score(y_t, y_p)

            metrics_by_type[atk_type] = {
                "acc": acc,
                "precision": report.get("1", {}).get("precision", 0.0),
                "recall": report.get("1", {}).get("recall", 0.0),
                "f1-score": report.get("1", {}).get("f1-score", 0.0),
                "support": len(y_t)
            }

            if verbose:
                print(f"\n=== Attack Type {atk_type} ===")
                print(f"Accuracy: {acc:.4f}")
                print(classification_report(y_t, y_p, zero_division=0))
                print("Confusion Matrix:")
                print(confusion_matrix(y_t, y_p))

        return metrics_by_type

    def visualize(self, epoch):
        self.net.eval()

        with torch.no_grad():
            benign_X = []
            benign_y = []
            mal_X = []
            mal_y = []

            for inputs, labels in self.test_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                encode, _ = self.net(inputs)

                list_label = labels.tolist()
                list_encode = encode.tolist()

                for i in range(len(list_label)):
                    if list_label[i] == 1:
                        mal_X.append(list_encode[i][0])
                        mal_y.append(list_encode[i][1])
                    else:
                        benign_X.append(list_encode[i][0])
                        benign_y.append(list_encode[i][1])

            plt.figure()
            plt.scatter(mal_X, mal_y, c="red", marker="x")
            plt.scatter(benign_X, benign_y, c="blue", marker="o")
            plt.savefig(
                "./visual/{}/epoch_{}_{}.pdf".format(
                    self.model_type, epoch, self.model_type
                )
            )
            plt.close()

    def visualize_z(self, epoch, pdf_writer):
        """
        Visualize latent z using t-SNE and save to a PDF.

        :param epoch: Current epoch
        :param pdf_writer: PdfPages writer to save the plot
        """

        self.net.eval()  # Set model to evaluation mode
        latent_z_list = []

        with torch.no_grad():
            for input, _ in self.train_data_loader:
                input = input.to(self.device)

                # Get latent representation
                encode, _ = self.net(input)
                latent_z_list.append(encode.cpu().numpy())

        # Concatenate all latent representations
        latent_z_array = np.concatenate(latent_z_list, axis=0)

        # Randomly select 200 samples if there are more than 200
        if len(latent_z_array) > 200:
            random_indices = np.random.choice(
                len(latent_z_array), size=200, replace=False
            )
            latent_z_array = latent_z_array[random_indices]

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        z_embedded = tsne.fit_transform(latent_z_array)

        # Plot t-SNE results
        plt.figure(figsize=(8, 8))
        plt.scatter(
            z_embedded[:, 0],
            z_embedded[:, 1],
            c="blue",
            label=f"Client {self.client_idx}",
        )
        plt.title(f"t-SNE of Latent z for Client {self.client_idx} at Epoch {epoch}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid()

        # Save to PDF
        pdf_writer.savefig()
        plt.close()
    
    
    def visualize_validate(self, epoch, threshold_re, threshold_z, auc, scores_re, scores_latent_z, labels):
        """
        Vẽ biểu đồ phân tán (scatter plot) với:
        - X: Latent Z (||z||^2)
        - Y: Reconstruction Error (RE)
        - Màu dựa theo nhãn (0: xanh, 1: đỏ)
        - Thêm đường threshold RE và Z
        """

        # Chỉ lấy tối đa 150 mẫu đầu tiên để trực quan
        num_samples = min(150, len(scores_re))
        scores_re = np.array(scores_re[:num_samples])
        scores_latent_z = np.array(scores_latent_z[:num_samples])
        labels = np.array(labels[:num_samples])

        # Tạo thư mục lưu nếu chưa có
        save_dir = "visual_unsw_test_thres"
        os.makedirs(save_dir, exist_ok=True)

        # Tên file
        file_name = f"{self.args.dataset}_{self.args.model_type}_{self.args.aggregation_type}_epoch{epoch}_client{self.client_idx}.png"
        file_path = os.path.join(save_dir, file_name)

        # Vẽ biểu đồ
        plt.figure(figsize=(8, 6))
        plt.title(
            f"Client {self.client_idx} | Epoch {epoch} | AUC: {auc:.6f}\nThresh_RE: {threshold_re:.6f} | Thresh_Z: {threshold_z:.6f}"
        )
        plt.xlabel("Latent Z (||z||^2)")
        plt.ylabel("Reconstruction Error (RE)")

        # Scatter points
        plt.scatter(
            scores_latent_z[labels == 0],
            scores_re[labels == 0],
            c="blue",
            s=12,
            label="Normal (0)",
            alpha=0.4
        )
        plt.scatter(
            scores_latent_z[labels == 1],
            scores_re[labels == 1],
            c="red",
            s=12,
            label="Anomalous (1)",
            alpha=0.4
        )

        # Threshold lines
        plt.axhline(y=threshold_re, color="black", linestyle="--", label="Threshold RE")
        plt.axvline(x=threshold_z, color="green", linestyle="--", label="Threshold Z")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
    
    def get_latent_and_labels(self, max_batches=None):
        """
        Trả về toàn bộ latent vectors và nhãn local.
        max_batches: giới hạn số batch để tiết kiệm memory (tùy chọn)
        """
        self.net.eval()
        z_list = []
        y_list = []

        with torch.no_grad():
            for i, (x, y) in enumerate(self.train_data_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                z, _ = self.net.forward(x)  # chỉ cần latent
                z_list.append(z)
                y_list.append(y)

                if max_batches is not None and i+1 >= max_batches:
                    break

        # concat toàn bộ batch
        z_all = torch.cat(z_list, dim=0)  # shape (total_samples, latent_dim)
        y_all = torch.cat(y_list, dim=0)  # shape (total_samples,)
        return z_all, y_all
