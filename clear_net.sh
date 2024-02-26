# if [ -z "$1" ]
# then
# 		echo 'Please input the exp trial_no to kill!'
# 		echo
# 		echo 'Ongoing trial processes for your reference: '
# 		ps -ef | grep `pwd` | grep 'ssh' | grep 'rank=0'
# 		exit
# fi
echo 'Process killed: '
# ps -ef | grep `pwd` | grep "[t]rial_no=$1"
# ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print $2}' | xargs kill -9
ps -ef | grep "nethogs" | awk '{print $2}' | xargs kill -9

clients="16.162.254.81 16.162.253.148 43.198.111.5 18.163.238.60"
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
for i in $clients; do
	# command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	command="ps -ef | grep "nethogs" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done