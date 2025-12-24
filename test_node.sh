#!/bin/bash

PARTITION="gpuA40x4"
RESERVATION="RH9"
GPUS=4

echo "===== SLURM Request Diagnostics ====="
echo "Partition    : $PARTITION"
echo "Reservation  : $RESERVATION"
echo "GPUs needed  : $GPUS"
echo

echo "----- Checking partition info -----"
scontrol show partition $PARTITION
echo

echo "----- Checking reservation info -----"
scontrol show reservation $RESERVATION
echo

echo "----- Nodes in partition -----"
sinfo -p $PARTITION -N -o "%N %T %G"
echo

echo "----- Nodes in reservation -----"
scontrol show reservation $RESERVATION | grep Nodes
echo

echo "----- Available nodes with >=${GPUS} GPUs -----"
sinfo -p $PARTITION -N -o "%N %T %G" | grep gpu | grep ":${GPUS}"
echo

echo "----- Checking node states -----"
sinfo -p $PARTITION -o "%N %T %R"
echo

echo "===== End Diagnostics ====="

