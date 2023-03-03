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


clients='18.163.112.214 18.162.149.191 16.163.141.88 16.162.45.130 18.166.225.51 18.162.111.137 16.163.146.64 18.162.111.65 18.163.182.179 18.166.178.80 18.166.30.86 43.198.86.162 18.163.182.123 18.162.154.99 18.163.124.20 16.163.143.211 18.163.114.212 16.162.45.177 18.167.20.19 18.166.62.148'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done