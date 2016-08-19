#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

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

void m128i_print(__m128i vVal)
{
	char *sVal;

	sVal = (char*)&vVal;
	for(int i = 0; i < 16; i++) {
		printf(" %02X ",sVal[i]);
	}
}

void fec_printbuf_start(uint8_t *buf, int blocksize, int matrix)
{
	__m128i val;
	for(int x = 0; x <= matrix; x++) {
		for(int y = 0; y <= matrix; y++) {
			for(int i = 0; i < blocksize; i+= 2048) {
				val = _mm_loadu_si128((__m128i*)&buf[y*(matrix+1)*blocksize+x*blocksize+i]);
				printf("FEC:INFO:X[%d],Y[%d],I[%4d]: ", x, y, i);
				m128i_print(val);
				printf("\n");
			}
		}
	}
}

// Row major 2D matrix
// Note2Self: X = Col, Y = Row
void fec_genfec(uint8_t *buf, int blocksize, int matrix)
{
	__m128i res, val;
	int iCurRowOffset;

	// Handle fec for each row of data blocks
	for(int y = 0; y < matrix; y++) {
		iCurRowOffset = y*(matrix+1)*blocksize;
		for(int i = 0; i < blocksize; i+= 16) {
			res = _mm_setzero_si128();
			for(int x = 0; x < matrix; x++) {
				val = _mm_loadu_si128((__m128i*)&buf[iCurRowOffset+x*blocksize+i]);
				res = _mm_xor_si128(res, val);
			}
			_mm_storeu_si128((__m128i*)&buf[iCurRowOffset+matrix*blocksize+i], res);
		}
	}
	// handle fec for each column of data blocks
	for(int x = 0; x < matrix; x++) {
		for(int i = 0; i < blocksize; i+= 16) {
			res = _mm_setzero_si128();
			for(int y = 0; y < matrix; y++) {
				val = _mm_loadu_si128((__m128i*)&buf[y*(matrix+1)*blocksize+x*blocksize+i]);
				res = _mm_xor_si128(res, val);
			}
			_mm_storeu_si128((__m128i*)&buf[matrix*(matrix+1)*blocksize+x*blocksize+i], res);
		}
	}
}

void fec_usefec(uint8_t *buf, int blocksize, int matrix)
{
	__m128i res, val;
	int iCurRowOffset;

	// Handle fec for each row of data blocks
	for(int y = 0; y <= matrix; y++) {
		iCurRowOffset = y*(matrix+1)*blocksize;
		for(int i = 0; i < blocksize; i+= 16) {
			res = _mm_setzero_si128();
			for(int x = 0; x <= matrix; x++) {
				val = _mm_loadu_si128((__m128i*)&buf[iCurRowOffset+x*blocksize+i]);
				res = _mm_xor_si128(res, val);
			}
			if (res != 0) {
				printf("FEC:WARN:Y=Row=%d Not Valid\n",y);
			}
		}
	}
	// handle fec for each column of data blocks
	for(int x = 0; x <= matrix; x++) {
		for(int i = 0; i < blocksize; i+= 16) {
			res = _mm_setzero_si128();
			for(int y = 0; y <= matrix; y++) {
				val = _mm_loadu_si128((__m128i*)&buf[y*(matrix+1)*blocksize+x*blocksize+i]);
				res = _mm_xor_si128(res, val);
			}
			if (res != 0) {
				printf("FEC:WARN:X=Col=%d Not Valid\n",x);
			}
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

	memset(buf, 0, FEC_BUFFERSIZE);
	hFSrc = open(argv[1],O_RDONLY);
	hFDst = open(argv[2],O_CREAT | O_WRONLY);

	iMatrix = 0;
	while(1) {
		printf("FEC:INFO: processing data matrix [%d]\n", iMatrix);
		if (fec_loadbuf(buf, hFSrc, FEC_BLOCKSIZE, FEC_DATAMATRIX1D) != 0) {
			printf("FEC:INFO: failed loading of data matrix [%d], maybe EOF, quiting...\n", iMatrix);
			break;
		} else {
			printf("FEC:INFO: loaded data matrix [%d]\n", iMatrix);
		}
		fec_printbuf_start(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D);
		fec_genfec(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D);
		fec_printbuf_start(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D);
		write(hFDst, buf, FEC_BUFFERSIZE);
		fec_usefec(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D);
		iMatrix += 1;
	}
	return 0;
}

