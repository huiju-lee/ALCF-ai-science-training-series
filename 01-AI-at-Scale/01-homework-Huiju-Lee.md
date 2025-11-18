# Homework
## 1. The counting of ranks, does not necessarily has to be a mix-and-match between mpi4py and PALS. 
## Try to implement the rank counting method using just PALS or mpi4py. device_count() methods can be useful here.

## Used just mpi4py method for the rank counting.
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

## 2. Play with different dimensions of the src and tgt tensors.
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
