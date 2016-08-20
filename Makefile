

fec: fec.c
	gcc -Wall -g -mrdseed -o fec fec.c

clean: fec
	rm -i fec

