#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h>

#define FEC_BLOCKSIZE 4096
#define FEC_DATAMATRIX1D 8
#define FEC_FULLMATRIX ((FEC_DATAMATRIX1D+1)*(FEC_DATAMATRIX1D+1))

int fec_validmeta(int blocksize, int matrix)
{
	if ((blocksize % 16) != 0) {
		return -1;
	}
	return 0;
}

void fec_genfec(uint8_t *buf, int blocksize, int matrix)
{
	__m128i res, val;
	int iCurRowOffset;

	for(int y = 0; y < matrix; y++) {
		iCurRowOffset = y*(matrix+1)*blocksize;
		for(int i = 0; i < blocksize; i+= 16) {
			res = _mm_setzero_si128();
			for(int x = 0; x < matrix; x++) {
				val = _mm_loadu_si128(&buf[iCurRowOffset+x*blocksize+i*16]);
				res = _mm_xor_si128(res, val);
			}
			_mm_storeu_si128(&buf[iCurRowOffset+matrix*blocksize+i*16], res);
		}
	}
}


int main(int argc, char **argv)
{

	uint8_t buf[FEC_FULLMATRIX*FEC_BLOCKSIZE];

	fec_genfec(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D);
	return 0;
}
