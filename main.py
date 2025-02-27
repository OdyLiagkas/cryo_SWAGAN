import subprocess
import sys
import argparse

def run_training(n_gpu, batch_size, lmdb_path):     
  
  # WILL HAVE TO CHANGE IT TO:
  
  #torchrun --nproc_per_node=1 --master_port=8981 train.py --arch swagan --batch 64 ../LMDB_100K   
  #because of depreciation of torch.distributed  ...
  
  command = [
        "python", "-m", "torch.distributed.launch", 
        f"--nproc_per_node={n_gpu}",  # Number of GPUs
        #f"--master_port={port}",  # Port for distributed training
        "train.py", 
        "--arch", "swagan",  # Specify the SWAGAN architecture
        "--batch", str(batch_size),  # Batch size
        lmdb_path  # Path to the LMDB dataset
    ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":

    n_gpu=1 # Number of GPUs to use for distributed training
    
    port=8981 # Master port for distributed training

    batch_size = 64 # Batch size for training

    imdb_path='../LMDB_100K'  # Path to LMDB dataset


    # Run training with provided arguments
    run_training(n_gpu, port, batch_size, lmdb_path)
