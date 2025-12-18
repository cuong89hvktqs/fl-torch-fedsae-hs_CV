# 📊 FLOW ANALYSIS: main.py → Các Class Khác

## 🔄 Luồng Code Chính

```
main.py 
  ↓
run_exp(config)
  ↓
Arguments(logger, config) 
  ↓
create_clients() + ServerClass()
  ↓
run_machine_learning()
```

---

## 🔍 Chi Tiết Kiểm Tra Tham Số

### 1️⃣ **main.py → config dict**

**File**: [main.py](main.py)

```python
config = {
    "dataset": args.d,
    "train_batch_size": args.tbs,
    "val_batch_size": args.vbs,
    "test_batch_size": 1,  # ✅ Cứng trong config
    "mal_batch_size": args.tbs,
    "dimension": args.di,
    "epochs": args.ep,
    "model_type": args.m,
    "noise_type": args.nt,
    "num_of_poisoned_workers": args.pw,      # ⚠️ LƯU Ý: tên key
    "poisoned_sample_ratio": args.pr,
    "learning_rate": args.lr,
    "noise_std": args.ns,
    "attack_noise_std": args.ans,
    "aggregation_type": args.agg,
    "coef_shrink_ae": args.cs,
    "threshold_multiplier": args.tm,
    "num_multi_class_clients": args.mc,
    "by_attack_type": args.at,
    "noniid": args.noniid
}
```

---

### 2️⃣ **config → Arguments class**

**File**: [function/arguments.py](function/arguments.py) (lines 1-50)

```python
class Arguments:
    def __init__(self, logger, config):
        # ✅ ĐÚNG - Tất cả key từ config được map đúng
        self.dataset = config["dataset"]
        self.train_batch_size = config["train_batch_size"]
        self.val_batch_size = config["val_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.mal_batch_size = config["mal_batch_size"]
        self.dimension = config["dimension"]
        self.epochs = config["epochs"]
        self.model_type = config["model_type"]
        self.noise_type = config["noise_type"]
        
        # ⚠️ LƯU Ý: tên key là "num_of_poisoned_workers" nhưng biến là "num_poisoned_workers"
        self.num_poisoned_workers = config["num_of_poisoned_workers"]  
        
        self.poisoned_sample_ratio = config["poisoned_sample_ratio"]
        self.learning_rate = config["learning_rate"]
        self.std_noise = config["noise_std"]
        self.attack_std_noise = config["attack_noise_std"]
        self.aggregation_type = config["aggregation_type"]
        self.coef_shrink_ae = config["coef_shrink_ae"]
        self.threshold_multiplier = config["threshold_multiplier"]
        self.num_multi_class_clients = config["num_multi_class_clients"]
        self.by_attack_type = config["by_attack_type"]
        self.noniid = config["noniid"]
        
        # ⚙️ MẶC ĐỊNH NỘI BỘ
        self.num_workers = 20  # Tổng số client
        self.es_offset = 100   # Early stopping
        self.cuda = True
        self.shuffle = False
        self.loss_function = torch.nn.MSELoss
```

**✅ KIỂM ĐỊNH**: Tất cả tham số được mapping chính xác

---

### 3️⃣ **Arguments → create_clients() → ClientClass**

**File**: [core/client_factory.py](core/client_factory.py)

```python
def create_clients(args, train_loaders, val_loaders, test_loaders):
    clients = []
    ClientClass = get_client_class(args.model_type)  # ✅ Động tải Client
    
    for idx in range(args.num_workers):  # 10 clients được tạo
        clients.append(
            ClientClass(
                args,           # ✅ Truyền toàn bộ args object
                idx,
                train_loaders[idx],
                val_loaders[idx],
                test_loaders[idx],
            )
        )
    return clients
```

**File**: [agents/clients/__init__.py](agents/clients/__init__.py)

```python
def get_client_class(model_type):
    model_map = {
        "AE": "clientAE",
        "DAE": "clientDAE",
        "SAE": "clientSAE",
        "FedMSE": "clientFedMSE",
        "SAE1": "clientSAE1",
        "SDAE": "clientSDAE",
        "SDAE1": "clientSDAE1",
        "SupAE": "clientSupAE",
        "clientDualLossAE": "clientDualLossAE",  # ⚠️ LỖI: key không trùng với model_type
        "MultiLossAE": "clientMultiLossAE",
        "MultiZAE": "clientMultiZAE",
        "PTL": "clientPTL"
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    module_name = model_map[model_type]
    class_name = "Client" + model_type  # ✅ Tự động build tên class
    module = __import__(f"agents.clients.{module_name}", fromlist=[class_name])
    ClientClass = getattr(module, class_name)
    return ClientClass
```

**✅ Ví dụ**: 
- Input: `args.model_type = "MultiZAE"`
- Output: `ClientMultiZAE` từ `agents/clients/clientMultiZAE.py`

---

### 4️⃣ **ClientClass.__init__() kiểm tra**

**File**: [agents/clients/clientMultiZAE.py](agents/clients/clientMultiZAE.py) (lines 1-70)

```python
class ClientMultiZAE:
    def __init__(self, args, client_idx, train_data_loader, val_data_loader, test_data_loader):
        self.args = args  # ✅ Lưu toàn bộ args
        self.client_idx = client_idx
        self.model_type = self.args.model_type  # ✅ Truy cập từ args.model_type
        
        self.device = self.initialize_device()
        self.set_net(self.load_default_model())
        
        self.loss_function = self.args.loss_function()  # ✅ MSELoss()
        self.optimizer = optim.Adam(
            self.net.parameters(), 
            lr=self.args.learning_rate  # ✅ Truy cập learning_rate từ args
        )
        
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
```

**✅ KIỂM ĐỊNH**: Tất cả tham số từ args được sử dụng đúng

---

### 5️⃣ **Arguments → ServerClass**

**File**: [agents/servers/__init__.py](agents/servers/__init__.py)

```python
def get_server_class(model_type):
    model_map = {
        "AE": "serverAE",
        "DAE": "serverDAE",
        "SAE": "serverSAE",
        "FedMSE": "serverFedMSE",
        "SAE1": "serverSAE1",
        "SDAE": "serverSDAE",
        "SDAE1": "serverSDAE1",
        "SupAE": "serverSupAE",
        "DualLossAE": "serverDualLossAE",  # ✅ Đúng
        "MultiLossAE": "serverMultiLossAE",
        "MultiZAE": "serverMultiZAE",
        "PTL": "serverPTL"
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unsupported model type for server: {model_type}")
    
    module_name = model_map[model_type]
    class_name = "Server" + model_type
    module = __import__(f"agents.servers.{module_name}", fromlist=[class_name])
    ServerClass = getattr(module, class_name)
    return ServerClass
```

**✅ Ví dụ**: 
- Input: `args.model_type = "MultiZAE"`
- Output: `ServerMultiZAE` từ `agents/servers/serverMultiZAE.py`

---

## 🐛 LỖI PHÁT HIỆN

### ❌ **LỖI #1: Client factory map sai**

**File**: [agents/clients/__init__.py](agents/clients/__init__.py)

```python
model_map = {
    ...
    "clientDualLossAE": "clientDualLossAE",  # ❌ LỖI!
    ...
}
```

**Vấn đề**: Key là `"clientDualLossAE"` nhưng main.py truyền `model_type="DualLossAE"`
- Khi gọi: `python main.py -m DualLossAE ...`
- → `"DualLossAE"` không tìm thấy trong model_map
- → **ValueError: Unsupported model type: DualLossAE**

**Sửa**:
```python
"DualLossAE": "clientDualLossAE",  # ✅ Đúng
```

---

### ❌ **LỖI #2: Không consistency giữa main.py evaluate.py**

**main.py**:
```python
parser.add_argument("-pw", type=int, default=0, ...)  # Tên arg là "-pw"
# Nhưng trong config map là: "num_of_poisoned_workers"
```

**evaluate.py**:
```python
parser.add_argument("-pw", "--poisoned_workers", ...)
# Nhưng khác name: args_ns.poisoned_workers
```

**Sửa**: Nên thống nhất tên biến

---

## ✅ KIỂM ĐỊNH CHUNG

| Thành phần | Trạng thái | Ghi chú |
|-----------|-----------|---------|
| main.py → config | ✅ Đúng | Tất cả tham số mapping chính xác |
| config → Arguments | ✅ Đúng | Tất cả key được map chính xác |
| Arguments → Clients | ✅ Đúng | args object được truyền đầy đủ |
| Client.__init__() | ✅ Đúng | Tất cả tham số từ args được sử dụng |
| Client factory (AE, SAE, MultiZAE, etc) | ✅ Đúng | Dynamic import hoạt động |
| **Client factory (DualLossAE)** | ❌ LỖI | Key map sai "clientDualLossAE" |
| Server factory | ✅ Đúng | Tất cả model được map đúng |
| Tham số Early Stopping | ✅ Đúng | args.es_offset = 100 mặc định |
| Tham số Loss function | ✅ Đúng | MSELoss được set mặc định |

---

## 🔧 **KHUYẾN NGHỊ FIX**

**1. Fix client factory map**:
```python
# agents/clients/__init__.py - Line 8
"DualLossAE": "clientDualLossAE",  # Sửa từ "clientDualLossAE"
```

**2. Thống nhất tên argument**:
```python
# main.py - Sử dụng "poisoned_workers" thay vì "num_of_poisoned_workers"
# hoặc thống nhất qua cả evaluate.py
```

**3. Thêm validation**:
```python
# main.py hoặc function/arguments.py
if args.num_workers < args.num_multi_class_clients:
    raise ValueError("Multi-class clients không thể vượt quá tổng clients")
```

