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
workers='18.166.51.62 18.166.51.62 18.163.55.176 18.163.55.176 16.163.55.57 16.163.55.57 43.198.31.5 43.198.31.5 18.166.76.65 18.166.76.65 18.166.54.10 18.166.54.10 18.162.55.73 18.162.55.73 18.162.155.75 18.162.155.75 16.163.50.241 16.163.50.241 16.162.25.13 16.162.25.13'
for i in $workers; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done