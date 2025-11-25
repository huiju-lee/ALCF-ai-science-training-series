#!/bin/bash -l
#PBS -A ALCFAITP
#PBS -l select=2
#PBS -N producer-consumer
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -j oe
#PBS -l place=scatter
#PBS -q debug

cd $PBS_O_WORKDIR

module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/ALCFAITP/03-Coupling-Sim-AI/_ai4s_simAI
export TMPDIR=/tmp
export PATH=$PATH:/opt/pbs/bin

# Set inputs
NUM_SIMS=128
GRID_SIZE=512

cd /home/huijulee/advanced-topics-in-ai-schedule/ALCF-ai-science-training-series/03-Coupling-Sim-AI/producer-consumer/

echo "Running producer-consumer scripts"
echo "with $NUM_SIMS simulations of $GRID_SIZE x $GRID_SIZE grid"
echo "Current working directory:" $(pwd)

# Warmup first
echo "Let's do a warmup run first with the Parsl + file system implementation (DISCARD THIS DATA)"
python /home/huijulee/advanced-topics-in-ai-schedule/ALCF-ai-science-training-series/03-Coupling-Sim-AI/producer-consumer/6_parsl_fs_producer_consumer.py --grid_size $GRID_SIZE --num_sims $NUM_SIMS
echo

# Run the tests
echo "Running with Parsl writing to the file system"
python /home/huijulee/advanced-topics-in-ai-schedule/ALCF-ai-science-training-series/03-Coupling-Sim-AI/producer-consumer/6_parsl_fs_producer_consumer.py --grid_size $GRID_SIZE --num_sims $NUM_SIMS
echo

echo "Running with DragonHPC"
dragon /home/huijulee/advanced-topics-in-ai-schedule/ALCF-ai-science-training-series/03-Coupling-Sim-AI/producer-consumer/8_dragon_producer_consumer.py --grid_size $GRID_SIZE --num_sims $NUM_SIMS
echo

echo "Running with Parsl tranfering data through futures (last since it will take longer)"
python /home/huijulee/advanced-topics-in-ai-schedule/ALCF-ai-science-training-series/03-Coupling-Sim-AI/producer-consumer/5_parsl_fut_producer_consumer.py --grid_size $GRID_SIZE --num_sims $NUM_SIMS
echo
