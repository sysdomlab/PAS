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

clients='18.166.55.130 18.167.85.7 18.163.111.195 18.166.63.24 16.162.55.143 43.198.71.196 18.167.68.174 18.163.35.5 18.162.214.72 16.163.57.106 18.163.111.48 18.167.103.121 18.166.55.75 43.198.71.241 16.163.52.171 18.162.134.98 16.163.42.208 18.166.65.224 18.166.74.175 16.162.86.7'
for i in $clients; do
	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
	ssh $i $command
done