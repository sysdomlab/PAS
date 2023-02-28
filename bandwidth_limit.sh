#!/usr/bin/env bash
home=$HOME'/CASP-code'
clients='18.162.144.155 16.162.188.0 43.198.79.184 18.166.69.254 18.162.149.82 43.198.11.120 18.162.210.45 43.198.14.96 16.163.145.139 16.162.137.238 43.198.12.20 18.162.207.246 16.162.122.205 18.166.75.17 43.198.84.151 18.162.149.71 18.162.141.170 16.162.118.114 43.198.83.15 18.163.28.157'

log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

bandwidth=1024
num=0
for i in $clients; do
    command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $bandwidth -d $bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done