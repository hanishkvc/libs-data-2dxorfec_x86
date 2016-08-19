#include <stdio.h>

#define FEC_BLOCKSIZE 4096
#define FEC_DATAMATRIX1D 8
#define FEC_FULLMATRIX ((FEC_DATAMTRIX1D+1)*(FEC_DATAMTRIX1D+1))

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
	int iCurRowOffset;

	for(y = 0; y < matrix; y++) {
		iCurRowOffset = y*(matrix+1)*blocksize;
		for(i = 0; i < blocksize; i+= 16) {
			res = 0;
			for(x = 0; x < matrix; x++) {
				val = __mm_loadu_si128(buf[iCurRowOffset+x*blocksize+i*16]);
				res = __mm_xor_si128(res, val);
			}
			__mm_storeu_si128(buf[iCurRowOffset+matrix*blocksize+i*16], res);
		}
	}
}


int main(int argc, char **argv)
{

	buf[FEC_FULLMATRIX*FEC_BLOCKSIZE];

	fec_genfec(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D);
	return 0;
}
