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

clients='18.162.144.155 16.162.188.0 43.198.79.184 18.166.69.254 18.162.149.82 43.198.11.120 18.162.210.45 43.198.14.96 16.163.145.139 16.162.137.238 43.198.12.20 18.162.207.246 16.162.122.205 18.166.75.17 43.198.84.151 18.162.149.71 18.162.141.170 16.162.118.114 43.198.83.15 18.163.28.157'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done