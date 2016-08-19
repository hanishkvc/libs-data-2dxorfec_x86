#include <stdio.h>

#define FEC_BLOCKSIZE 4096
#define FEC_MATRIX 8

int fec_validmeta(int blocksize, int matrix)
{
	if ((blocksize % 16) != 0) {
		return -1;
	}
	return 0;
}

void fec_genfec(uint8 *buf, int blocksize, int matrix)
{
	__m128i res, val;

	for(y = 0; y < matrix; y++) {
		for(i = 0; i < blocksize; i+= 16) {
			res = 0;
			for(x = 0; x < matrix; x++) {
				val = __mm_loadu_si128(buf[y*blocksize*(matrix+1)+x*blocksize+i*16]);
				res = __mm_xor_si128(res, val);
			}
			__mm_storeu_si128(buf[y*blocksize*(matrix+1)+matrix*blocksize+i*16], res);
		}
	}
}

