# Homework
### 1. The counting of ranks, does not necessarily has to be a mix-and-match between mpi4py and PALS. Try to implement the rank counting method using just PALS or mpi4py. device_count() methods can be useful here.

- Used just mpi4py method for the rank counting.
```
# DDP: Set environmental variables used by PyTorch
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
num_gpus = torch.cuda.device_count()
LOCAL_RANK = RANK % num_gpus
#LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID')
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.polaris.alcf.anl.gov"
os.environ['MASTER_PORT'] = str(2345)
print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}.{MASTER_ADDR}")
```

### 2. Play with different dimensions of the src and tgt tensors.
```
src = torch.rand((2048, 1, 512))
tgt = torch.rand((2048, 20, 512))
```
total train time: 4.62s
```
src = torch.rand((4096, 1, 512))
tgt = torch.rand((4096, 20, 512))
```
total train time: 8.99s
```
src = torch.rand((8192, 1, 512))
tgt = torch.rand((8192, 20, 512))
```
total train time: 17.86s
```
src = torch.rand((2048, 4, 512))
tgt = torch.rand((2048, 128, 512))
```
total train time: 48.88s

### 3. Explore the cost of collective communication, by setting up a scenario, where you have only two ranks, but each rank resides on a different node. Profile and try to reason about the results.

- Scenario A — Both ranks on the SAME NODE
```
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
let NRANKS=${NNODES}*${NRANKS_PER_NODE}
N=2
PPN=2
NODES=1
```
x3109c0s37b0n0.hsn.cm.polaris.alcf.anl.gov 0: cpubind:list x3109c0s37b0n0 pid 2105784 rank 0 0: mask 0x3
x3109c0s37b0n0.hsn.cm.polaris.alcf.anl.gov 1: cpubind:list x3109c0s37b0n0 pid 2105785 rank 1 1: mask 0x300
x3109c0s37b0n0.hsn.cm.polaris.alcf.anl.gov 0: DDP: Hi from rank 0 of 2 with local rank 0.x3109c0s37b0n0
x3109c0s37b0n0.hsn.cm.polaris.alcf.anl.gov 1: DDP: Hi from rank 1 of 2 with local rank 1.x3109c0s37b0n0
total train time: 10.84s

<img width="1024" height="752" alt="image" src="https://github.com/user-attachments/assets/bc7b218a-1fa0-49b5-8215-e2dcf06906a9" />

- Scenario B — Ranks on DIFFERENT NODES
```
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
let NRANKS=${NNODES}*${NRANKS_PER_NODE}
N=2
PPN=1
NODES=2
```
x3002c0s31b1n0.hsn.cm.polaris.alcf.anl.gov 0: cpubind:list x3002c0s31b1n0 pid 3581078 rank 0 0: mask 0x3
x3002c0s37b0n0.hsn.cm.polaris.alcf.anl.gov 1: cpubind:list x3002c0s37b0n0 pid 883424 rank 1 0: mask 0x3
x3002c0s31b1n0.hsn.cm.polaris.alcf.anl.gov 0: DDP: Hi from rank 0 of 2 with local rank 0.x3002c0s31b1n0
x3002c0s37b0n0.hsn.cm.polaris.alcf.anl.gov 1: DDP: Hi from rank 1 of 2 with local rank 0.x3002c0s31b1n0
total train time: 38.63s

<img width="1003" height="720" alt="image" src="https://github.com/user-attachments/assets/20641756-36dd-42ba-a352-59e15b977e56" />

Inter-node NCCL all-reduce is ~3.5× slower
This clearly demonstrates that collective communication becomes a major bottleneck when scaling training across nodes.

### 4. Try other file formats to explore the I/O bottleneck.
- hpf5 file
  <img width="994" height="711" alt="image" src="https://github.com/user-attachments/assets/41f5f908-52ea-4d6e-97d1-0a689f742edd" />
  Data loader duration: 3ms 552µs 611ns
  total train time: 11.02s
  
- .pt file
  <img width="993" height="719" alt="image" src="https://github.com/user-attachments/assets/f6afe9d5-6a9d-4da7-85b3-20a9e85fa42e" />
  Data loader duration: 926µs 587ns
  total train time: 10.90s

Even though HDF5 per-batch I/O is about 3× slower, the overall training time differs very little because the dataset is small and the compute cost of the Transformer dominates the runtime.


