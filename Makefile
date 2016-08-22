

fec: fec.c
	gcc -Wall -g -mrdseed -mpopcnt -o fec fec.c

clean:
	rm -i fec || /bin/true

allclean: clean
	rm -i test.bin* || /bin/true

prepare:
	dd if=/dev/zero of=test.bin.datazero bs=4096 count=64
	dd if=/dev/urandom of=test.bin.datarandom bs=4096 count=64
	ln -s test.bin.datarandom test.bin.data

testgen: fec
	./fec gen test.bin.data test.bin.datafec

testuse: fec
	./fec use test.bin.datafec test.bin.datarecovered

testall: fec
	./fec all test.bin.data test.bin.datafec.all

run: fec
	./fec test.bin test.bin.res
