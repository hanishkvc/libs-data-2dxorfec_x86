#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define FEC_BLOCKSIZE 4096
#define FEC_DATAMATRIX1D 8
#define FEC_FULLMATRIX ((FEC_DATAMATRIX1D+1)*(FEC_DATAMATRIX1D+1))
#define FEC_BUFFERSIZE (FEC_FULLMATRIX*FEC_BLOCKSIZE)

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
				val = _mm_loadu_si128((__m128i*)&buf[iCurRowOffset+x*blocksize+i*16]);
				res = _mm_xor_si128(res, val);
			}
			_mm_storeu_si128((__m128i*)&buf[iCurRowOffset+matrix*blocksize+i*16], res);
		}
	}
}

int fec_loadbuf(uint8_t *buf, int hFile, int blocksize, int matrix)
{
	int iRead;
	int iCurRowOffset;
	int iBufSizeDataRow = matrix*blocksize;
	for(int y = 0; y < matrix; y++) {
		iCurRowOffset = y*(matrix+1)*blocksize;
		iRead = read(hFile, &buf[iCurRowOffset], iBufSizeDataRow);
		if (iRead != iBufSizeDataRow)
			return -1;
	}
	return 0;
}

int main(int argc, char **argv)
{

	uint8_t buf[FEC_BUFFERSIZE];

	int hFSrc, hFDst;
	int iMatrix;

	hFSrc = open(argv[1],O_RDONLY);
	hFDst = open(argv[2],O_WRONLY);

	iMatrix = 0;
	while(1) {
		if (fec_loadbuf(buf, hFSrc, FEC_BLOCKSIZE, FEC_DATAMATRIX1D) != 0) {
			break;
		} else {
			printf("FEC:INFO: processing data matrix [%d]\n", iMatrix);
		}
		fec_genfec(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D);
		write(hFDst, buf, FEC_BUFFERSIZE);
	}
	return 0;
}

