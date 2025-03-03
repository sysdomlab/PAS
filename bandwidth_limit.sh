#!/usr/bin/env bash
home=$HOME'/CASP-code'

clients="192.168.0.12 192.168.0.11 192.168.0.18 192.168.0.15 192.168.0.10 192.168.0.17 192.168.0.13 192.168.0.24 192.168.0.38 192.168.0.16 192.168.0.34 192.168.0.25 192.168.0.29 192.168.0.42 192.168.0.14 192.168.0.20 192.168.0.44 192.168.0.22 192.168.0.23 192.168.0.43 192.168.0.19 192.168.0.30 192.168.0.28 192.168.0.21 192.168.0.40 192.168.0.46 192.168.0.33 192.168.0.36 192.168.0.31 192.168.0.45 192.168.0.27 192.168.0.26 192.168.0.35 192.168.0.32 192.168.0.39 192.168.0.37 192.168.0.47 192.168.0.49 192.168.0.48 192.168.0.41"

log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

up_bandwidth=896
down_bandwidth=896
num=0
for i in $clients; do
    command="sudo $HOME/wondershaper/wondershaper -a eth0 -u $up_bandwidth -d $down_bandwidth"
    # command="sudo $HOME/wondershaper/wondershaper -c -a eth0"
    # command="speedtest"
    ssh $i $command > $log_dir/worker_$num.log 2>&1
    num=$((num+1))
done