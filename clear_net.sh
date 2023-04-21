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

clients="43.198.101.204 16.162.252.241 18.162.133.190 18.162.156.192"
# clients='43.198.24.242 43.198.72.11 43.198.77.77 43.198.22.171'
for i in $clients; do
	# command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	command="ps -ef | grep "nethogs" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done