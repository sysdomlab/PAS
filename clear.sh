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


clients='16.162.24.129 18.166.62.132 43.198.86.93 16.162.34.42 43.198.14.118 18.162.112.54 18.166.226.217 16.163.149.8 18.162.149.135 16.162.45.172 16.163.142.94 18.166.29.120 18.163.114.13 18.162.112.109 16.163.155.86 43.198.11.132 16.162.118.131 16.163.149.227 18.166.215.47 16.162.119.231'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done