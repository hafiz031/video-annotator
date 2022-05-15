mkdir lol; 
cd lol;

for i in {1..100}
	do
	touch file_${i}.txt
	echo "Hello file_${i}" > file_${i}.txt
	done;

cd ..;
