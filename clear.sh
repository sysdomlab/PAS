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

workers='18.162.155.234 18.162.45.16 18.163.116.86 18.166.176.205 16.162.47.226 18.162.154.167 43.198.13.129 18.163.56.89 18.162.230.55 16.163.140.148 43.198.15.247 18.166.29.93 16.162.109.34 16.163.129.15 18.162.110.27 18.163.112.138 18.162.47.90 18.162.167.16 43.198.81.107 18.163.7.117'

for i in $workers; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done