echo "sync"
if [ ! -n "$1" ]; then
exit
fi
echo "OK"
workdir=$HOME'/CASP-code'
for i in $other_workers
do
	ssh $i "sudo rm -rf $workdir/$1"
	scp -r $1 "$i:$workdir/$1"
	echo $i 
done
