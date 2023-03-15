#!/usr/bin/env bash
home=$HOME'/CASP-code'

clients='18.166.67.56 16.163.58.32 18.166.65.15 18.167.96.144 18.166.77.173 18.167.109.117 16.163.56.102 18.166.67.59 18.166.70.7 18.162.126.68 18.163.5.183 18.166.67.217 16.163.109.65 18.162.116.225 43.198.77.24 18.167.102.181 18.162.143.142 18.163.181.40 18.167.116.234 18.163.105.192'
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

up_bandwidth=5120
down_bandwidth=10240
num=0
for i in $clients; do
    # command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $up_bandwidth -d $down_bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    command="speedtest"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done