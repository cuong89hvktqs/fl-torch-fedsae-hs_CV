import numpy as np
import torch

from collections import defaultdict

def distribute_batches_equally(data_loader, num_workers):
    """
    Gives each worker the same number of batches of data.

    :param data_loader: Data loader
    :type data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(data_loader):
        worker_idx = batch_idx % num_workers

        distributed_dataset[worker_idx].append((data, target))

    return distributed_dataset

def distribute_non_iid_dirichlet(
    data_loader, num_workers, alpha=0.5, normal_label=0
):
    """
    - Normal label chia theo Dirichlet nhưng đảm bảo mỗi client có >= 1 mẫu normal.
    - Attack labels chia Dirichlet như bình thường.
    - Giữ nguyên cấu trúc batch của data_loader.
    """

    # =========================
    # B1. Gộp toàn bộ dữ liệu
    # =========================
    all_x, all_y = [], []
    for X_batch, y_batch in data_loader:
        all_x.append(X_batch)
        all_y.append(y_batch)
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    # =========================
    # B2. Gom index theo class
    # =========================
    class_indices = defaultdict(list)
    for idx, y in enumerate(all_y):
        class_indices[int(y.item())].append(idx)

    unique_labels = sorted(class_indices.keys())
    attack_labels = [c for c in unique_labels if c != normal_label]

    normal_idxs = np.array(class_indices[normal_label])
    np.random.shuffle(normal_idxs)

    # ============================================
    # B3. Chia Normal theo Dirichlet (non-iid)
    #     NHƯNG ép mỗi client có ≥ 1 mẫu
    # ============================================
    normal_props = np.random.dirichlet([alpha] * num_workers)
    normal_counts = (normal_props * len(normal_idxs)).astype(int)

    # Sửa rounding: đảm bảo tổng khớp
    diff = len(normal_idxs) - normal_counts.sum()
    for _ in range(diff):
        normal_counts[np.argmax(normal_props)] += 1

    # Ép mỗi client có ≥ 1 mẫu normal
    # Nếu client nào có 0 → lấy bớt từ client nhiều nhất
    for w in range(num_workers):
        if normal_counts[w] == 0:
            donor = np.argmax(normal_counts)
            normal_counts[donor] -= 1
            normal_counts[w] += 1

    # === Tạo index normal cho từng client ===
    worker_indices = [[] for _ in range(num_workers)]
    start = 0
    for w in range(num_workers):
        end = start + normal_counts[w]
        worker_indices[w].extend(normal_idxs[start:end].tolist())
        start = end

    # ============================================
    # B4. Phân phối Attack theo Dirichlet bình thường
    # ============================================
    if len(attack_labels) > 0:
        attack_dir = np.random.dirichlet([alpha] * num_workers, len(attack_labels))

        for label_id, proportions in zip(attack_labels, attack_dir):
            idx_list = np.array(class_indices[label_id])
            np.random.shuffle(idx_list)

            total = len(idx_list)
            cls_split = (proportions * total).astype(int)

            diff = total - cls_split.sum()
            for _ in range(diff):
                cls_split[np.argmax(proportions)] += 1

            # add vào workers
            start = 0
            for w in range(num_workers):
                end = start + cls_split[w]
                worker_indices[w].extend(idx_list[start:end].tolist())
                start = end

    # =========================
    # B5. Shuffle mỗi client
    # =========================
    for w in range(num_workers):
        np.random.shuffle(worker_indices[w])

    # =========================
    # B6. Chia theo batch size
    # =========================
    batch_size = data_loader.batch_size
    distributed_dataset = [[] for _ in range(num_workers)]

    for w in range(num_workers):
        idxs = worker_indices[w]
        for i in range(0, len(idxs), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            if len(batch_idxs) == 0:
                continue
            Xb = all_x[batch_idxs]
            yb = all_y[batch_idxs]
            distributed_dataset[w].append((Xb, yb))

    return distributed_dataset

def distribute_non_iid_dirichlet_zero_lable_client_bak(
        data_loader,
        num_workers,
        alpha=0.5,
        num_zero_label_clients=0,
        zero_label=0):

    """
    Đảm bảo:
    1. Pure clients (0..num_zero_label_clients-1):
       - CHỈ CÓ zero_label
       - nhận nhiều zero_label nhất
       - non-IID (Dirichlet)

    2. Multi-class clients:
       - có zero_label (bắt buộc)
       - zero_label ít hơn pure
       - non-IID nhẹ (Dirichlet)

    3. Attack labels:
       - CHỈ phân phối cho multi-clients
       - non-IID mạnh
       - số loại tấn công per client khác nhau
    """

    from collections import defaultdict
    import numpy as np
    import torch

    # === B1. Gom toàn bộ batch ===
    all_x, all_y = [], []
    for Xb, yb in data_loader:
        all_x.append(Xb)
        all_y.append(yb)
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    # Gom index theo nhãn
    class_indices = defaultdict(list)
    for idx, y in enumerate(all_y):
        class_indices[int(y.item())].append(idx)

    worker_indices = [[] for _ in range(num_workers)]
    unique_labels = list(class_indices.keys())

    # ============================================================
    # B2 — Chia ZERO LABEL
    # ============================================================
    zero_idxs = np.array(class_indices[zero_label])
    np.random.shuffle(zero_idxs)

    # ------------------------
    # PURE CLIENTS: chỉ nhận ZERO LABEL
    # ------------------------
    if num_zero_label_clients > 0:
        pure = num_zero_label_clients

        pure_portion = int(0.6 * len(zero_idxs))   # pure clients lấy nhiều nhất
        pure_slice = zero_idxs[:pure_portion]

        # non-IID mạnh
        pure_dist = np.random.dirichlet([3] * pure)
        pure_split = (pure_dist * len(pure_slice)).astype(int)

        diff = len(pure_slice) - pure_split.sum()
        if diff > 0:
            pure_split[np.argmax(pure_split)] += diff

        start = 0
        for i in range(pure):
            end = start + pure_split[i]
            worker_indices[i].extend(pure_slice[start:end].tolist())
            start = end

        leftover_zero = zero_idxs[pure_portion:]

    else:
        leftover_zero = zero_idxs

    # ------------------------
    # MULTI CLIENTS: còn lại vẫn phải có zero_label
    # ------------------------
    num_multi = max(1, num_workers - num_zero_label_clients)

    multi_dist = np.random.dirichlet([6.0] * num_multi)
    multi_split = (multi_dist * len(leftover_zero)).astype(int)

    diff = len(leftover_zero) - multi_split.sum()
    if diff > 0:
        multi_split[np.argmax(multi_split)] += diff

    # đảm bảo mỗi multi-client có ≥ 1 zero_label
    for i in range(num_multi):
        if multi_split[i] == 0:
            donor = np.argmax(multi_split)
            if multi_split[donor] > 1:
                multi_split[donor] -= 1
                multi_split[i] += 1

    start = 0
    for i in range(num_multi):
        w = num_zero_label_clients + i
        end = start + multi_split[i]
        worker_indices[w].extend(leftover_zero[start:end].tolist())
        start = end

    # ============================================================
    # B3 — ATTACK LABELS (NHÃN ≠ ZERO)
    # -> CHỈ phân phối cho multi clients
    # -> PURE CLIENTS hoàn toàn không nhận
    # ============================================================
    attack_labels = [l for l in unique_labels if l != zero_label]

    attack_dirichlet = np.random.dirichlet([alpha] * num_multi, len(attack_labels))

    for k, label_id in enumerate(attack_labels):

        idxs = np.array(class_indices[label_id])
        np.random.shuffle(idxs)
        total = len(idxs)

        props = attack_dirichlet[k]
        split = (props * total).astype(int)

        diff = total - split.sum()
        if diff > 0:
            split[np.argmax(props)] += diff

        start = 0
        for i in range(num_multi):
            w = num_zero_label_clients + i   # chỉ multi clients
            end = start + split[i]
            worker_indices[w].extend(idxs[start:end].tolist())
            start = end

    # ============================================================
    # B4 — Shuffle từng client
    # ============================================================
    for w in range(num_workers):
        np.random.shuffle(worker_indices[w])

    # ============================================================
    # B5 — Build lại batch theo batch_size ban đầu
    # ============================================================
    batch_size = data_loader.batch_size
    distributed = [[] for _ in range(num_workers)]

    for w in range(num_workers):
        idxs = worker_indices[w]
        for i in range(0, len(idxs), batch_size):
            batch = idxs[i:i+batch_size]
            distributed[w].append((all_x[batch], all_y[batch]))

    return distributed


def distribute_non_iid_dirichlet_zero_lable_client(
        data_loader,
        num_workers,
        alpha=0.5,
        num_zero_label_clients=0,
        zero_label=0):

    """
    Đảm bảo:
    1. Pure clients:
       - CHỈ có zero_label
       - nhận nhiều zero_label nhất
       - non-IID mạnh

    2. Multi clients:
       - có zero_label (ít hơn pure)
       - số LOẠI nhãn tấn công khác nhau (1..K)
       - attack labels non-IID mạnh

    3. Attack labels:
       - mỗi client chỉ có 1 tập con nhãn tấn công
       - client khác nhau có số loại attack khác nhau
    """

    from collections import defaultdict
    import numpy as np
    import torch

    # ============================================================
    # B1 — Gom toàn bộ dữ liệu
    # ============================================================
    all_x, all_y = [], []
    for Xb, yb in data_loader:
        all_x.append(Xb)
        all_y.append(yb)
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    class_indices = defaultdict(list)
    for idx, y in enumerate(all_y):
        class_indices[int(y.item())].append(idx)

    worker_indices = [[] for _ in range(num_workers)]
    unique_labels = list(class_indices.keys())

    # ============================================================
    # B2 — ZERO LABEL (GIỮ NGUYÊN LOGIC CỦA BẠN)
    # ============================================================
    zero_idxs = np.array(class_indices[zero_label])
    np.random.shuffle(zero_idxs)

    if num_zero_label_clients > 0:
        pure = num_zero_label_clients
        pure_portion = int(0.6 * len(zero_idxs))
        pure_slice = zero_idxs[:pure_portion]

        pure_dist = np.random.dirichlet([3] * pure)
        pure_split = (pure_dist * len(pure_slice)).astype(int)

        diff = len(pure_slice) - pure_split.sum()
        pure_split[np.argmax(pure_split)] += diff

        start = 0
        for i in range(pure):
            end = start + pure_split[i]
            worker_indices[i].extend(pure_slice[start:end].tolist())
            start = end

        leftover_zero = zero_idxs[pure_portion:]
    else:
        leftover_zero = zero_idxs

    num_multi = max(1, num_workers - num_zero_label_clients)

    multi_dist = np.random.dirichlet([6.0] * num_multi)
    multi_split = (multi_dist * len(leftover_zero)).astype(int)

    diff = len(leftover_zero) - multi_split.sum()
    multi_split[np.argmax(multi_split)] += diff

    for i in range(num_multi):
        if multi_split[i] == 0:
            donor = np.argmax(multi_split)
            multi_split[donor] -= 1
            multi_split[i] += 1

    start = 0
    for i in range(num_multi):
        w = num_zero_label_clients + i
        end = start + multi_split[i]
        worker_indices[w].extend(leftover_zero[start:end].tolist())
        start = end

    # ============================================================
    # B3 — ATTACK LABELS (SỐ LOẠI NHÃN NON-IID)
    # ============================================================
    attack_labels = [l for l in unique_labels if l != zero_label]
    K = len(attack_labels)

    multi_clients = list(range(num_zero_label_clients, num_workers))

    # ---- 1. Quyết định số loại attack per client (1..K)
    # phân bố lệch để non-IID
    attack_count_dist = np.random.dirichlet([0.3] * num_multi)
    attack_counts = np.clip(
        (attack_count_dist * K).astype(int) + 1,
        1,
        K
    )

    # ---- 2. Gán tập nhãn attack cho từng client
    client_attack_map = {}
    for i, w in enumerate(multi_clients):
        client_attack_map[w] = set(
            np.random.choice(
                attack_labels,
                size=attack_counts[i],
                replace=False
            )
        )

    # ---- 3. Với mỗi nhãn attack, tìm client được phép nhận
    for label in attack_labels:
        idxs = np.array(class_indices[label])
        np.random.shuffle(idxs)

        eligible_clients = [
            w for w in multi_clients
            if label in client_attack_map[w]
        ]

        if len(eligible_clients) == 0:
            continue

        # Dirichlet mạnh trong nhóm eligible
        dist = np.random.dirichlet([alpha] * len(eligible_clients))
        split = (dist * len(idxs)).astype(int)

        diff = len(idxs) - split.sum()
        split[np.argmax(split)] += diff

        start = 0
        for i, w in enumerate(eligible_clients):
            end = start + split[i]
            worker_indices[w].extend(idxs[start:end].tolist())
            start = end

    # ============================================================
    # B4 — Shuffle
    # ============================================================
    for w in range(num_workers):
        np.random.shuffle(worker_indices[w])

    # ============================================================
    # B5 — Build batch
    # ============================================================
    batch_size = data_loader.batch_size
    distributed = [[] for _ in range(num_workers)]

    for w in range(num_workers):
        idxs = worker_indices[w]
        for i in range(0, len(idxs), batch_size):
            batch = idxs[i:i + batch_size]
            distributed[w].append((all_x[batch], all_y[batch]))

    return distributed



#hàm nảy chỉ tạo ra các cleitn chỉ mới có các mẫu khác nhau và nhưng client nào cũng đều có tất cả các nhãn bất thường
def distribute_non_iid_dirichlet_V0(data_loader, num_workers, alpha=0.5):
    """
    Chia dữ liệu non-IID theo Dirichlet nhưng vẫn giữ nguyên STRUCTURE giống
    distribute_batches_equally:
        distributed_dataset[worker] = [(X_batch, y_batch), ...]
    """

    # === B1. Gộp toàn bộ batch thành 1 tensor lớn ===
    all_x = []
    all_y = []

    for X_batch, y_batch in data_loader:
        all_x.append(X_batch)
        all_y.append(y_batch)

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    # === B2. Gom các index của từng class ===
    class_indices = defaultdict(list)
    for idx, y in enumerate(all_y):
        class_indices[int(y.item())].append(idx)

    unique_labels = list(class_indices.keys())
    num_classes = len(unique_labels)

    # === B3. Sinh phân phối Dirichlet (num_classes x num_workers) ===
    dirichlet_dist = np.random.dirichlet([alpha] * num_workers, num_classes)

    # === B4. Phân chia index theo Dirichlet cho từng worker ===
    worker_indices = [[] for _ in range(num_workers)]

    for class_id, proportions in zip(unique_labels, dirichlet_dist):
        idx_list = np.array(class_indices[class_id])
        np.random.shuffle(idx_list)

        # số mẫu class này
        total = len(idx_list)
        cls_split = (proportions / proportions.sum() * total).astype(int)

        # chỉnh lại tổng
        diff = total - cls_split.sum()
        if diff > 0:
            cls_split[np.argmax(cls_split)] += diff

        # phân phối cho từng worker
        start = 0
        for w in range(num_workers):
            end = start + cls_split[w]
            worker_indices[w].extend(idx_list[start:end].tolist())
            start = end

    # === B5. Trộn index mỗi worker để tránh block same-class ===
    for w in range(num_workers):
        np.random.shuffle(worker_indices[w])

    # === B6. Chia thành batch (giống hệt distribute_batches_equally) ===
    distributed_dataset = [[] for _ in range(num_workers)]
    batch_size = data_loader.batch_size

    for w in range(num_workers):
        idxs = worker_indices[w]

        for i in range(0, len(idxs), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            X_batch = all_x[batch_idxs]
            y_batch = all_y[batch_idxs]
            distributed_dataset[w].append((X_batch, y_batch))

    return distributed_dataset

#hàm nảy chỉ tạo ra các cleitn chỉ mới có các mẫu khác nhau và nhưng client nào cũng đều có tất cả các nhãn bất thường
def distribute_non_iid_dirichlet_zero_lable_client_V0(
        data_loader,
        num_workers,
        alpha=0.5,
        num_pure_label_clients=0,
        pure_label=0):
    """
    Chia dữ liệu non-IID theo Dirichlet nhưng:
        - Các client [0 ... num_pure_label_clients-1] chỉ chứa nhãn pure_label

    Output format giữ nguyên:
        distributed_dataset[w] = [(X_batch, y_batch), ...]
    """

    from collections import defaultdict
    import torch
    import numpy as np

    # === B1. Gom toàn bộ batch ===
    all_x = []
    all_y = []
    for X_batch, y_batch in data_loader:
        all_x.append(X_batch)
        all_y.append(y_batch)

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    # === B2. Gom index theo từng nhãn ===
    class_indices = defaultdict(list)
    for idx, y in enumerate(all_y):
        class_indices[int(y.item())].append(idx)

    unique_labels = list(class_indices.keys())
    num_classes = len(unique_labels)

    # === B3. Chuẩn bị phân phối Dirichlet cho các client còn lại ===
    num_remaining = num_workers - num_pure_label_clients
    dirichlet_dist = np.random.dirichlet([alpha] * num_remaining, num_classes)

    # === B4. Khởi tạo danh sách index cho từng client ===
    worker_indices = [[] for _ in range(num_workers)]
    # ============================
    # B5: FIX LỖI — CHIA PURE CLIENT NGẪU NHIÊN
    # ============================
    pure_idxs = np.array(class_indices[pure_label])
    np.random.shuffle(pure_idxs)
    total_pure = len(pure_idxs)

    # random phân phối theo Dirichlet để chia 100% pure label
    pure_dist = np.random.dirichlet([1.0] * num_pure_label_clients)
    pure_split = (pure_dist * total_pure).astype(int)

    # fix rounding
    diff = total_pure - pure_split.sum()
    if diff > 0:
        pure_split[np.argmax(pure_split)] += diff

    # gán pure index
    start = 0
    for w in range(num_pure_label_clients):
        end = start + pure_split[w]
        worker_indices[w].extend(pure_idxs[start:end].tolist())
        start = end

    # số pure index còn lại chia cho clients còn lại
    leftover_pure = pure_idxs[start:]
    # ================================
    # B6. CHIA CÁC CLASS CHO CÁC CLIENT CÒN LẠI
    # ================================
    for label_id in unique_labels:

        idx_list = np.array(class_indices[label_id])
        np.random.shuffle(idx_list)

        # Nếu là pure_label → dùng leftover
        if label_id == pure_label:
            idx_list = leftover_pure

        total = len(idx_list)
        if total == 0:
            continue

        proportions = dirichlet_dist[label_id]
        cls_split = (proportions / proportions.sum() * total).astype(int)

        # fix rounding
        diff = total - cls_split.sum()
        if diff > 0:
            cls_split[np.argmax(cls_split)] += diff

        start = 0
        for wi in range(num_pure_label_clients, num_workers):
            end = start + cls_split[wi - num_pure_label_clients]
            worker_indices[wi].extend(idx_list[start:end].tolist())
            start = end

    # === B7. Shuffle từng client ===
    for w in range(num_workers):
        np.random.shuffle(worker_indices[w])

    # === B8. Chia lại thành batch giống distribute_batches_equally ===
    distributed_dataset = [[] for _ in range(num_workers)]
    batch_size = data_loader.batch_size

    for w in range(num_workers):
        idxs = worker_indices[w]
        for i in range(0, len(idxs), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            X_batch = all_x[batch_idxs]
            y_batch = all_y[batch_idxs]
            distributed_dataset[w].append((X_batch, y_batch))

    return distributed_dataset




def apply_data_replacement(X, Y, mal_data_loader, replacement_ratio):
    """
    Replace class labels using the replacement method

    :param X: data features
    :type X: numpy.Array()
    :param Y: data labels
    :type Y: numpy.Array()
    :param mal_data_loader: Mal data
    :type mal_data_loader: DataLoader
    :param replacement_ratio: Ratio to update sample
    :type replacement_method: float
    """
    X_mal = np.array(
        [tensor.numpy() for batch in mal_data_loader for tensor in batch[0]]
    )

    mal_id = np.random.randint(
        len(X_mal), size=round(len(X) * replacement_ratio))
    X_id = np.random.randint(len(X), size=round(len(X) * replacement_ratio))

    for id in range(len(mal_id)):
        X[X_id[id]] = X_mal[mal_id[id]]

    return (X, Y)


def apply_data_noise(X, Y, attack_noise_std, replacement_ratio):
    """
    Replace class labels using the replacement method

    :param X: data features
    :type X: numpy.Array()
    :param Y: data labels
    :type Y: numpy.Array()
    :param replacement_ratio: Ratio to update sample
    :type replacement_method: float
    """

    for id in range(round(len(X) * replacement_ratio)):
        X[id] = torch.add(
            torch.Tensor(X[id]),
            (attack_noise_std**0.5) * torch.rand_like(torch.Tensor(X[id])),
        )
    return (X, Y)


def poison_data(
    logger,
    distributed_dataset,
    num_workers,
    poisoned_worker_ids,
    noise_type,
    mal_data_loader,
    replacement_ratio,
    attack_noise_std,
):
    """
    Poison worker data

    :param logger: logger
    :type logger: loguru.logger
    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param num_workers: Number of workers overall
    :type num_workers: int
    :param poisoned_worker_ids: IDs poisoned workers
    :type poisoned_worker_ids: list(int)
    :param noise_type: type of noise (label_flipping or gaussian_noise)
    :type noise_type: string
    :param mal_data_loader: malicious data
    :type mal_data_loader: data loader
    :param replacement_method: Replacement methods to use to replace
    :type replacement_method: list(method)
    """
    poisoned_dataset = []
    logger.info("Poisoning data for workers: {}".format(
        str(poisoned_worker_ids)))
    # class_labels = list(set(distributed_dataset[0][1]))
    for worker_idx in range(num_workers):
        if worker_idx in poisoned_worker_ids:
            if noise_type == "label_flipping":
                poisoned_dataset.append(
                    apply_data_replacement(
                        distributed_dataset[worker_idx][0],
                        distributed_dataset[worker_idx][1],
                        mal_data_loader,
                        replacement_ratio,
                    )
                )
            else:
                poisoned_dataset.append(
                    apply_data_noise(
                        distributed_dataset[worker_idx][0],
                        distributed_dataset[worker_idx][1],
                        attack_noise_std,
                        replacement_ratio,
                    )
                )
        else:
            poisoned_dataset.append(distributed_dataset[worker_idx])
    return poisoned_dataset


def convert_distributed_data_into_numpy(distributed_dataset):
    """
    Converts a distributed dataset (returned by a data distribution method) from Tensors into numpy arrays.

    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    """
    converted_distributed_dataset = []

    for worker_idx in range(len(distributed_dataset)):
        worker_data = distributed_dataset[worker_idx]

        X_ = np.array(
            [tensor.numpy()
             for batch in worker_data for tensor in batch[0]]
        )
        Y_ = np.array(
            [tensor.numpy()
             for batch in worker_data for tensor in batch[1]]
        )

        converted_distributed_dataset.append((X_, Y_))

    return converted_distributed_dataset
