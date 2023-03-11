#!/usr/bin/env bash
home=$HOME'/CASP-code'

clients='18.166.67.81 18.167.84.47 18.162.214.77 16.163.58.59 16.163.126.159 16.163.103.144 16.163.55.148 43.198.78.224 43.198.78.62 18.163.123.17 18.163.77.138 18.163.183.231 18.166.51.241 18.166.55.213 18.167.121.232 18.167.85.67 18.166.55.177 43.198.76.107 18.166.53.92 18.163.35.255'
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

up_bandwidth=3072
down_bandwidth=3072
num=0
for i in $clients; do
    # command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $up_bandwidth -d $down_bandwidth && speedtest"
    command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    # command="speedtest"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done