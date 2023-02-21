#!/usr/bin/env bash
home=$HOME'/CASP-code'
clients='18.162.55.161 18.166.67.172 18.163.77.191 16.163.40.18 18.163.50.185 18.162.116.86 18.163.73.147 18.162.114.6 43.198.24.254 16.163.103.16 18.163.129.160 43.198.74.74 18.163.8.137 18.166.78.151 18.166.54.95 18.167.109.223 18.166.51.117 18.163.121.81 43.198.19.64 18.163.123.206'

log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

bandwidth=5120
num=0
for i in $clients; do
    command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $bandwidth -d $bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done