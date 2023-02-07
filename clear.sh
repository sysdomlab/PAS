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

workers='16.163.55.246 18.163.84.174 18.166.71.35 16.162.106.7 18.162.48.187 18.166.75.148 43.198.16.173 43.198.13.209 18.162.152.136 16.163.106.124 16.162.87.196 43.198.86.121 43.198.11.127 16.162.105.36 16.162.22.184 18.162.168.223 18.166.31.49 18.163.195.198 16.162.45.244 18.162.230.3'

for i in $workers; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done