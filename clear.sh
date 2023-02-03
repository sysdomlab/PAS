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
# ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print $2}' | xargs kill -9
workers='18.162.126.237 18.167.99.80 16.163.99.154 18.163.46.222'
for i in $workers; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done