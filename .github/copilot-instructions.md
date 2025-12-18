# AI Copilot Instructions for FL-Torch-FedSAE-HS

## Project Overview
This is a **Federated Learning (FL) framework for anomaly detection** using autoencoders (SAE, MultiZAE, DualLossAE, etc.) with Byzantine robustness against poisoned clients. The system trains distributed clients locally then aggregates models server-side using multiple strategies.

## Architecture Overview

### Core Data Flow
1. **Data Loading** ([function/datasets/](function/datasets/)) → 2. **Client Distribution** ([core/data_processing.py](core/data_processing.py)) → 3. **FL Training Loop** ([core/experiment_runner.py](core/experiment_runner.py)) → 4. **Server Aggregation** ([agents/servers/](agents/servers/)) → 5. **Evaluation** ([evaluate.py](evaluate.py))

### Key Components

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **Models** | Autoencoder variants for anomaly detection | [function/nets/](function/nets/) (AE, VAE, MultiZAE, DualLossAE, SupAE) |
| **Clients** | Local training agents, one per worker | [agents/clients/](agents/clients/) — named `client{ModelType}.py` |
| **Servers** | Aggregation & global model updates | [agents/servers/](agents/servers/) — named `server{ModelType}.py` |
| **Data Processing** | IID/Non-IID distribution, poisoning simulation | [core/data_processing.py](core/data_processing.py) & [function/utils/data_loader_util.py](function/utils/data_loader_util.py) |
| **Config** | Dataset & hyperparameter specifications | [config/](config/) (e.g., `unsw.yaml`, `cic_ids.yaml`) |

## Critical Patterns & Conventions

### Model-Specific Factories (Dynamic Dispatch)
- **Client classes** are loaded dynamically via [agents/clients/__init__.py](agents/clients/__init__.py) using `get_client_class(model_type)` 
- **Server classes** loaded via [agents/servers/__init__.py](agents/servers/__init__.py) using `get_server_class(model_type)`
- Model type (e.g., "MultiZAE", "AE") is passed via CLI argument `-m` and maps to module names like `clientMultiZAE.py`
- **Adding a new model requires**: (1) Create `function/nets/newmodel.py`, (2) Add `ClientNewModel` & `ServerNewModel` classes, (3) Update both factory `__init__.py` files

### Client-Server Communication Pattern
1. **Client training** ([agents/clients/clientXXX.py](agents/clients/clientXXX.py)):
   - `train(epoch)` → local MSELoss on reconstruction & latent dimension loss
   - `validate(epoch)` → sets `threshold_re` (reconstruction) and `threshold_z` (latent)
   - `get_nn_parameters()` → returns model weights for aggregation
   - `load_nn_parameters(weights)` → loads aggregated weights

2. **Server aggregation** ([agents/servers/serverXXX.py](agents/servers/serverXXX.py)):
   - `train_on_clients(epoch, clients)` → orchestrates client.train() and validation
   - Aggregates via `average_nn_parameters()`, `attention_average_nn_parameters()`, or `kmeans_cluster_parameters()`
   - Early stopping: clients stop training if no improvement after `args.es_offset` epochs (default: 100)

### Multi-Class vs Single-Class Client Splits
- [core/experiment_runner.py](core/experiment_runner.py) lines 73-87 split clients:
  - **Single-class clients** (indices 0 to `num_single-1`): see normal data only
  - **Multi-class clients** (indices `num_single` to `total_clients-1`): see normal + attack data
  - Controlled by CLI arg `-mc <num_multi_class_clients>`
  - Poisoned clients chosen within each group

### Data Distribution Modes
- **IID** (default `args.noniid=False`): `distribute_batches_equally()` — equal splits
- **Non-IID** (`-noniid True`): `distribute_non_iid_dirichlet()` with alpha=0.5; single-class clients get only label-0 via `distribute_non_iid_dirichlet_zero_lable_client()`
- Binary labels (0=normal, 1=attack) remapped at [core/experiment_runner.py](core/experiment_runner.py) lines 62-64 EXCEPT for MultiZAE (multi-attack support)

### Poisoning & Byzantine Attacks
- Simulated via [function/utils/data_loader_util.py](function/utils/data_loader_util.py):
  - **label_flipping** (default): flips labels in `mal_data_loader` (attack samples)
  - **gaussian_noise** (alternative): adds Gaussian noise to samples
  - Ratio controlled by `-pr <poisoned_sample_ratio>` (default: 1.0 = all samples in poisoned client poisoned)
  - Noise std: `-ns <noise_std>` (default: 0.001) for noise perturbation

## Essential Workflows

### Running Experiments
```bash
# 1. Initialize dataset (one-time per dataset)
python initialize_env.py -data <dataset>

# 2. Run training (example: MultiZAE on UNSW)
python main.py -d unsw -m MultiZAE -tbs 128 -vbs 1 -di 196 -lr 0.001 -ep 1000 \
  -agg average -nt label_flipping -pw 0 -pr 1 -ns 0.001 -ans 3 -cs 1 -tm 0 -mc 5

# 3. Evaluate on test set
python evaluate.py -d unsw -m MultiZAE -tbs 128 -vbs 128 -di 196 -ep 1000 \
  --model_dir saved_models/MultiZAE/5/average/unsw \
  --log_csv logs/MultiZAE/unsw_MultiZAE_
```

**Critical Args** (see [main.py](main.py)):
- `-d`: dataset (unsw, cic_ids, nb_iot, nsl_kdd, wsn_ds, etc.) → config/unsw.yaml
- `-m`: model (AE, DAE, SAE, MultiZAE, DualLossAE, SupAE, PTL, FedMSE)
- `-pw`: poisoned clients count (0=no attack), `-pr`: poison ratio in poisoned client
- `-nt`: noise type (label_flipping, gaussian_noise)
- `-agg`: aggregation (average, attention, cluster, kmeans)
- `-mc`: multi-class clients count (attack-aware clients)
- `-tm`: threshold multiplier for anomaly detection threshold

### Adding a New Autoencoder Model
1. Implement `function/nets/newmodel.py` with class `NewModel(nn.Module)` inheriting standard forward/encode/decode methods
2. Create `agents/clients/clientNewModel.py` with `ClientNewModel` class (copy [clientMultiZAE.py](agents/clients/clientMultiZAE.py) as template)
3. Create `agents/servers/serverNewModel.py` with `ServerNewModel` class (copy [serverMultiZAE.py](agents/servers/serverMultiZAE.py) as template)
4. Register in [agents/clients/__init__.py](agents/clients/__init__.py) and [agents/servers/__init__.py](agents/servers/__init__.py) model maps
5. Test: `python main.py -m NewModel -d unsw ...`

### Debugging & Logging
- Logs written via `loguru` to files in [logs/](logs/) directory (auto-generated per experiment)
- Log file format: `{dataset}_{model}_{epochs}_{pr}_{pw}_{ans}_{agg}_{tm}_{mc}_{timestamp}.log`
- Saved models stored in [saved_models/](saved_models/) with structure: `{model}/{num_multi_class_clients}/{agg_type}/{dataset}/`
- Best checkpoint saved when validation loss improves; periodic checkpoints every 10 epochs

### Configuration from YAML
[config/](config/) files define multi-value experiment grids (e.g., `unsw.yaml`):
- Multiple model types, learning rates, aggregation methods can be listed
- [initialize_env.py](initialize_env.py) expands grid & generates data loaders for each dataset
- Use config to batch experiments with consistent hyperparameters

## Cross-Component Integration Points
- **CLI parsing** ([main.py](main.py)) → **Arguments setup** ([function/arguments.py](function/arguments.py)) → **Experiment runner** ([core/experiment_runner.py](core/experiment_runner.py))
- **Data distribution** uses filtered DataLoaders from [function/datasets/dataset.py](function/datasets/dataset.py)
- **Thresholding** logic in client validate() feeds to server for anomaly detection tuning
- **Visualization** via [visualize.py](visualize.py) calls `visualize_muitizAE()` for multi-attack analysis

## Project-Specific Conventions Differ from Standard FL
1. **Asymmetric client roles**: single-class vs multi-class clients by design (not random)
2. **Reconstruction-based detection**: focuses on reconstruction error + latent dimension, not classification
3. **Two-tier thresholding**: separate thresholds for reconstruction (`threshold_re`) and latent space (`threshold_z`)
4. **Early stopping per client**: independent stopping per worker (not global synchronization)
5. **IID vs Non-IID on same clients**: non-IID mode forces single-class clients to label-0 only

## Dataset Expectations
- CSV format with normalized features (0-1)
- Last column = binary label (0=normal, 1=attack) OR multi-class attack types
- Placed in [data/{dataset}/](data/) directory as `{dataset}_Train.csv` and `{dataset}_Test.csv`
- Dimension (`-di`) must match feature count in CSV
- Datasets supported: unsw, cic_ids, nb_iot, nsl_kdd, wsn_ds, spambase, ton_iot_*, ctu13_08
