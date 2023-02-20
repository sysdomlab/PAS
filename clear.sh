if [ -z "$1" ]
then
		echo 'Please input the exp trial_no to kill!'
		echo
		echo 'Ongoing trial processes for your reference: '
		ps -ef | grep `pwd` | grep 'ssh' | grep 'rank=0'
		exit
fi
echo 'Process killed: '
ps -ef | grep `pwd` | grep "[t]rial_no=$1"
ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print $2}' | xargs kill -9
ps -ef | grep "multiprocessing" | awk '{print $2}' | xargs kill -9


# workers='18.167.0.57 18.163.188.30 18.162.44.220 43.198.95.97 16.163.156.117 18.162.123.143 18.162.44.79 18.167.134.169 43.198.96.81 18.163.24.92 16.163.1.98 18.162.120.27 18.167.51.97 43.198.95.178 18.162.43.201 18.163.194.43 18.162.142.216 16.162.88.146 43.198.90.67 18.167.165.118'

# for i in $workers; do
# 	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
# 	ssh $i $command
# 	command="ps -ef | grep 'multiprocessing' | awk '{print \$2}' | xargs kill -9"
# 	ssh $i $command
# done