# PAS: Towards Accurate and Efficient Federated Learning with Parameter-Adaptive Synchronization

[![Paper: IWQoS 2024](https://img.shields.io/badge/Paper-IWQoS%202024-blue)](https://ieeexplore.ieee.org/document/10682875)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**License:** This project is licensed under the [MIT License](LICENSE).

[Introduction and Citation](#introduction-and-citation) · [Project Structure](#project-structure) · [Installation and Usage](#installation-and-usage) · [Contact](#contact)

## Introduction and Citation

This repository contains the federated learning experiment code for PAS. It uses PyTorch for local model training and RPyC for parameter exchange and server aggregation. The implementation includes parameter-level synchronization period adjustment and adaptive parameter freezing to study the tradeoff between training accuracy and communication overhead.

The method is named **PAS** in the paper. The project is published as `sysdomlab/PAS`. Its implementation and launch arguments retain the names `CAS` and `CAS-APF`. `CAS` adjusts synchronization periods for individual parameters, while `CAS-APF` adds parameter freezing. The repository also includes experiment entry points for `FedAvg`, `FedProx`, `AdaComm`, `FedLAMA`, and `GIFT`.

**Paper:** [PAS: Towards Accurate and Efficient Federated Learning with Parameter-Adaptive Synchronization](https://ieeexplore.ieee.org/document/10682875)

**Authors:** Zuo Gan, Chen Chen, Jiayi Zhang, Gaoxiong Zeng, Yifei Zhu, Jieru Zhao, Quan Chen, Minyi Guo.

**Venue:** 2024 IEEE/ACM 32nd International Symposium on Quality of Service (IWQoS), pp. 1–10.

**DOI:** [10.1109/IWQoS61813.2024.10682875](https://doi.org/10.1109/IWQoS61813.2024.10682875). Bibliographic details are available in the [Crossref metadata](https://api.crossref.org/works/10.1109/IWQoS61813.2024.10682875).

If this project is useful for your research, please cite:

```bibtex
@inproceedings{gan2024pas,
  author    = {Zuo Gan and Chen Chen and Jiayi Zhang and Gaoxiong Zeng
               and Yifei Zhu and Jieru Zhao and Quan Chen and Minyi Guo},
  title     = {{PAS}: Towards Accurate and Efficient Federated Learning
               with Parameter-Adaptive Synchronization},
  booktitle = {2024 IEEE/ACM 32nd International Symposium on Quality of Service (IWQoS)},
  year      = {2024},
  pages     = {1--10},
  doi       = {10.1109/IWQoS61813.2024.10682875},
  url       = {https://ieeexplore.ieee.org/document/10682875}
}
```

## Project Structure

```text
PAS/
├── README.md                        # Overview, citation, and usage instructions
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
├── dataset_manager.py               # Data loading and client data partitioning
├── model_manager.py                 # Models, optimizers, flattened parameters, and checkpoints
├── models/                          # CNN, LSTM, ResNet, VGG, and other model implementations
├── expotential/                     # Experiment implementation used by the training script
│   ├── client/
│   │   ├── worker_process.py        # Local training entry point
│   │   ├── cas_sync_process.py      # Separate synchronization process for CAS / CAS-APF
│   │   ├── cas_manager.py           # Synchronization period adjustment and parameter freezing
│   │   ├── fedavg_manager.py        # FedAvg / FedProx synchronization manager
│   │   ├── adacomm_manager.py       # AdaComm synchronization manager
│   │   ├── fedlama_manager.py       # FedLAMA synchronization manager
│   │   └── gift_manager.py          # GIFT synchronization manager
│   └── server/
│       ├── server_process.py        # Aggregation server entry point
│       ├── server_test_process.py   # Global model evaluation service entry point
│       ├── cas_server.py            # CAS / CAS-APF aggregation
│       ├── fedavg_server.py         # FedAvg / FedProx aggregation
│       ├── adacomm_server.py        # AdaComm aggregation
│       ├── fedlama_server.py        # FedLAMA aggregation
│       └── gift_server.py           # GIFT aggregation
├── train.sh                         # Launch experiments across machines through SSH
├── sync.sh                          # Distribute a file or directory to clients
├── clear.sh                         # Terminate experiment processes by trial_no
├── bandwidth_limit.sh               # Set client bandwidth limits with wondershaper
└── clear_net.sh                     # Terminate nethogs network monitoring processes
```

`expotential` is the actual directory name in this repository. Use this spelling in configuration and launch commands.

## Installation and Usage

### 1. Set Up the Environment

The scripts for experiments across machines target Linux systems with Bash, SSH, and SCP. Use a Python 3.10 environment for the existing PyTorch 1.13 dependency, and install the same dependency versions on every participating node. For GPU training, install a PyTorch build compatible with the local driver; see the [PyTorch installation instructions for previous versions](https://pytorch.org/get-started/previous-versions/#v1130).

```bash
git clone https://github.com/sysdomlab/PAS.git
cd PAS

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy<2" "torch==1.13.0" "torchvision==0.14.0"
python -m pip install -r requirements.txt "numpy<2" "torchvision==0.14.0"
python -m pip check
```

`dataset_manager.py` and several model files directly import `torchvision`, so the commands explicitly install a version matching PyTorch. The `numpy<2` constraint avoids compatibility issues between older PyTorch releases and NumPy 2. These commands cover the basic image classification and KWS paths. Extensions such as BERT and ViT require additional dependencies, local model files, and corresponding code configuration.

Check that the core modules can be imported:

```bash
python -c "import torch, torchvision, rpyc; from model_manager import Model_Manager; from dataset_manager import Dataset_Manager; print('CUDA:', torch.cuda.is_available())"
```

### 2. Prepare the Data

Launch processes from the repository root. Dataset and checkpoint paths are resolved relative to the current working directory.

| Dataset argument | Example models | Data preparation |
| --- | --- | --- |
| `Mnist` | `CNN_Mnist` | Downloaded to `./Mnist/` on first use |
| `EMnist` | `CNN_EMnist` | The balanced split is downloaded to `./EMnist/` on first use |
| `FashionMnist` | `CNN_FashionMnist` | Downloaded to `./FashionMnist/` on first use |
| `Cifar10` | `CNN_Cifar10`, `ResNet20_Cifar10` | Downloaded to `./Cifar10/` on first use |
| `KWS` | `LSTM_KWS` | Manually prepare `./kws/index_train.txt`, `./kws/index_test.txt`, and feature files |

Each line in a KWS index file must contain `<feature_file_path> <integer_label>`, separated by a single space. Each feature file must be readable by `numpy.loadtxt` and contain 500 numeric values that can be reshaped to `(50, 10)`. Labels must correspond to the 10 classes. KWS data and feature preprocessing scripts are not included in this repository. The current defaults in `train.sh` are `LSTM_KWS` / `KWS`; prepare these files before running, or select an image classification configuration from the table.

The server and every client need access to their corresponding datasets. For the single-machine example, download MNIST in advance to avoid concurrent downloads:

```bash
python - <<'PY'
from torchvision import datasets

datasets.MNIST(root="./Mnist", train=True, download=True)
datasets.MNIST(root="./Mnist", train=False, download=True)
PY
```

### 3. Run a Single-Machine Example

This example uses `CNN_Mnist`, `Mnist`, `CAS-APF`, and two clients to demonstrate the startup sequence. Open six terminals, enter the repository root in each one, and activate the same environment. The example uses the CPU. For GPU runs, set `CUDA_VISIBLE_DEVICES` according to the available resources.

**Terminal 1: Start the evaluation service.**

```bash
CUDA_VISIBLE_DEVICES="" python expotential/server/server_test_process.py \
  --port=10000 --trial_no=0 --remarks=mnist_demo \
  --model=CNN_Mnist --dataset=Mnist
```

Wait for the evaluation service to finish initializing its model and dataset before starting the aggregation server.

**Terminal 2: Start the aggregation server.**

```bash
CUDA_VISIBLE_DEVICES="" python expotential/server/server_process.py \
  --port=20000 --world_size=2 --init_period=128 \
  --trial_no=0 --remarks=mnist_demo --interlayer_type=CAS-APF \
  --test_process_port=10000
```

**Terminals 3 and 4: Start a synchronization process for each client.**

```bash
# Terminal 3: rank 0
CUDA_VISIBLE_DEVICES="" python expotential/client/cas_sync_process.py \
  --server_ip=127.0.0.1 --server_port=20000 --rank=0 \
  --port=30000 --trial_no=0 --remarks=mnist_demo
```

```bash
# Terminal 4: rank 1
CUDA_VISIBLE_DEVICES="" python expotential/client/cas_sync_process.py \
  --server_ip=127.0.0.1 --server_port=20000 --rank=1 \
  --port=30010 --trial_no=0 --remarks=mnist_demo
```

**Terminals 5 and 6: Start the two training clients.**

```bash
# Terminal 5: rank 0
CUDA_VISIBLE_DEVICES="" python expotential/client/worker_process.py \
  --server_ip=127.0.0.1 --server_port=20000 --sync_process_port=30000 \
  --rank=0 --world_size=2 --trial_no=0 --remarks=mnist_demo \
  --model=CNN_Mnist --dataset=Mnist --interlayer_type=CAS-APF \
  --batchsize=100 --init_period=128 --dirichlet_alpha=0.5
```

```bash
# Terminal 6: rank 1
CUDA_VISIBLE_DEVICES="" python expotential/client/worker_process.py \
  --server_ip=127.0.0.1 --server_port=20000 --sync_process_port=30010 \
  --rank=1 --world_size=2 --trial_no=0 --remarks=mnist_demo \
  --model=CNN_Mnist --dataset=Mnist --interlayer_type=CAS-APF \
  --batchsize=100 --init_period=128 --dirichlet_alpha=0.5
```

Both training clients must join before the server can complete initial aggregation. Clients print local training and synchronization logs, while the evaluation service reports `round_id`, `accuracy`, and `loss`. Use this example to check process connections and the basic training workflow. Adjust experiment settings using the paper and your available resources. The current training loop allows up to 1,000,000 epochs; press `Ctrl+C` in each terminal to end the demonstration.

### 4. Run Experiments Across Machines

Prepare the repository and datasets at the same paths on every machine, and configure SSH access from the launch node to the server and clients. By default, `train.sh` calls `python3` through SSH. Ensure that `python3` in remote non-interactive sessions uses the environment with the required dependencies, or change the script to use that environment's absolute interpreter path.

Edit the experiment settings in `train.sh`:

| Setting | Description |
| --- | --- |
| Repository path | Defaults to `$HOME/CASP-code`; update this setting to the actual PAS deployment path on each node |
| `server` / `clients` | Aggregation server address and space-separated client addresses; the client count determines `world_size` |
| `model` / `dataset` | Select matching model and dataset names |
| `interlayer_type` | `CAS-APF`, `CAS`, `FedAvg`, `FedProx`, `AdaComm`, `FedLAMA`, or `GIFT` |
| `method` | Use `expotential`; directories for other commented-out methods are not included in this repository |
| `batchsize` / `init_period` | Local batch size and initial synchronization period; the server and clients must use the same synchronization period |
| `dirichlet_alpha` | Client data partitioning parameter; see `dataset_manager.py` for the sampling implementation |

CAS / CAS-APF training processes connect to their synchronization services through `localhost`, so each synchronization process must run on the same machine as its training client. The evaluation service and aggregation server must also run on the same machine.

Allow connections between nodes on the required ports. The script assigns ports using the experiment identifier `trial_no`: `10000 + trial_no` for evaluation, `20000 + trial_no` for aggregation, and `30000 + trial_no + 10 * rank` for client synchronization.

Once configuration is complete, launch from the repository root:

```bash
mkdir -p Logs
bash train.sh
```

When prompted, enter experiment notes and visible GPU IDs, such as `0` or `0,1`. Leave the GPU IDs empty for CPU experiments. Logs and code snapshots are saved to `Logs/<trial_no>_<remarks>/`, and `Latest_Log` points to the most recent experiment directory:

```bash
tail -f Latest_Log/server_process.log
tail -f Latest_Log/client_process_0.log
```

The script's fixed startup delays may be too short for initial dataset downloads. Prepare data on every node before launching. To terminate an experiment started by the script, first configure the client addresses in `clear.sh` for that experiment, then run `bash clear.sh <trial_no>`. Check that the server's evaluation and aggregation processes have also exited.

`sync.sh` distributes a specified file or directory after deleting the remote target with the same name. Check its node and path settings before use. For bandwidth experiments, configure the nodes, network interface, wondershaper path, and upload/download limits in `bandwidth_limit.sh`, then run the script.

## Contact

**Maintainer:** [@gz944367214](https://github.com/gz944367214).

For questions about the code, experiment reproduction, or bug reports, contact the maintainer through [GitHub Issues](https://github.com/sysdomlab/PAS/issues). Include your experiment configuration, dependency versions, and relevant logs.
