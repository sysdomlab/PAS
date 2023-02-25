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


# clients='16.162.102.20 18.162.168.86 43.198.86.242 18.166.154.157 16.162.100.184 16.163.138.198 18.166.31.181 18.162.52.54 18.162.49.84 16.162.188.65 18.166.31.175 18.163.37.39 43.198.85.223 18.166.214.136 43.198.86.8 18.167.36.91 16.163.155.174 16.162.163.239 16.162.163.160 18.162.152.107'

# for i in $clients; do
# 	command="ps -ef | grep `pwd` | grep "[t]rial_no=$1" | awk '{print \$2}' | xargs kill -9"
# 	ssh $i $command
# done