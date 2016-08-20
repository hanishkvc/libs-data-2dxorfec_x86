

fec: fec.c
	gcc -Wall -g -mrdseed -mpopcnt -o fec fec.c

clean: fec
	rm -i fec

