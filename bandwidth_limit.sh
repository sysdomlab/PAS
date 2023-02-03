#!/usr/bin/env bash
home=$HOME'/CASP-code'
master="16.162.44.221"
workers="16.162.44.221 43.198.73.133 16.163.58.81 18.166.74.29 43.198.72.65 18.163.111.51 16.162.44.42 18.163.121.128 18.163.189.253 43.198.25.217 18.162.148.84 18.163.41.8 18.162.236.153 18.162.55.205 43.198.76.251 16.163.50.171 18.167.72.138 16.163.58.65 16.163.58.54 18.163.180.171"

log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

bandwidth=5120
num=0
for i in $workers; do
    command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $bandwidth -d $bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done