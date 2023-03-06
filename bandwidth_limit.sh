#!/usr/bin/env bash
home=$HOME'/CASP-code'

clients='43.198.24.242 43.198.72.11 16.163.43.98 18.167.90.183 18.163.123.55 18.163.127.201 18.166.78.71 18.166.78.90 43.198.77.77 18.162.155.215 18.163.111.72 18.162.214.100 43.198.22.171 16.163.58.206 18.167.103.202 18.163.50.244 18.167.103.113 18.162.214.8 18.163.73.23 18.166.78.83'
log_dir=$home/Logs/bandwidth_limit
mkdir -p $log_dir

bandwidth=1024
num=0
for i in $clients; do
    command="sudo $HOME/wondershaper/wondershaper -a ens5 -u $bandwidth -d $bandwidth && speedtest"
    # command="sudo $HOME/wondershaper/wondershaper -c -a ens5"
    # command="speedtest"
    nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
    num=$((num+1))
done