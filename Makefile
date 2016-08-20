

fec: fec.c
	gcc -Wall -g -mrdseed -mpopcnt -o fec fec.c

clean: fec
	rm -i fec

allclean: clean
	rm -i test.bin*

prepare:
	dd if=/dev/zero of=test.bin bs=4096 count=64

run: fec
	./fec test.bin test.bin.res
