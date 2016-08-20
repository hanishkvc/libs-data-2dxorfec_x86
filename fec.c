#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h>
#include <x86intrin.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#define FEC_BLOCKSIZE 4096
#define FEC_DATAMATRIX1D 8
#define FEC_FULLMATRIX ((FEC_DATAMATRIX1D+1)*(FEC_DATAMATRIX1D+1))
#define FEC_BUFFERSIZE (FEC_FULLMATRIX*FEC_BLOCKSIZE)

#define FEC_MAXMATRIX1D 32

// this structure tells if a given block in the fec buffer is valid or not
// the bits with in each blocks[index] correspond to the columns with in that row.
// So for now this allows a 32x32 fec matrix (including both data and fec)
// NOTE: row and column are numbered starting from 0

struct fecMatrixFlag {
	uint32_t blocks[FEC_MAXMATRIX1D];
	uint32_t rowview[FEC_MAXMATRIX1D];
	uint32_t colview[FEC_MAXMATRIX1D];
	uint32_t row, col;
};

struct fecMatrixFlag gMatFlag;

void fecmatflag_blockset(struct fecMatrixFlag *flag, int rowy, int colx)
{
	flag->blocks[rowy] |= (1 << colx);
	flag->rowview[rowy] |= (1 << colx);
	flag->colview[colx] |= (1 << rowy);
}

void fecmatflag_blockclear(struct fecMatrixFlag *flag, int rowy, int colx)
{
	flag->blocks[rowy] &= ~(1 << colx);
	flag->rowview[rowy] &= ~(1 << colx);
	flag->colview[colx] &= ~(1 << rowy);
}

int fecmatflag_blockget(struct fecMatrixFlag *flag, int rowy, int colx)
{
	return (flag->blocks[rowy] & (1 << colx));
}

void fecmatflag_rowset(struct fecMatrixFlag *flag, int rowy)
{
	flag->row |= (1 << rowy);
}

void fecmatflag_colset(struct fecMatrixFlag *flag, int colx)
{
	flag->col |= (1 << colx);
}

void fecmatflag_print(struct fecMatrixFlag *flag, int dmatrix)
{
	printf("FEC:INFO:matflag:row[0x%08X], col[0x%08X]\n", flag->row, flag->col);
	for (int i = 0; i <= dmatrix; i++) {
		printf("FEC:INFO:blocks[row:%02d]=0x%08X\n", i, flag->blocks[i]);
		printf("FEC:INFO:   rowview[%02d]=0x%08X\n", i, flag->rowview[i]);
		printf("FEC:INFO:   colview[%02d]=0x%08X\n", i, flag->colview[i]);
	}
}

int fec_validmeta(int blocksize, int dmatrix)
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

void fec_printbuf_start(uint8_t *buf, int blocksize, int dmatrix)
{
	__m128i val;
	for(int colx = 0; colx <= dmatrix; colx++) {
		for(int rowy = 0; rowy <= dmatrix; rowy++) {
			for(int i = 0; i < blocksize; i+= 2048) {
				val = _mm_loadu_si128((__m128i*)&buf[rowy*(dmatrix+1)*blocksize+colx*blocksize+i]);
				printf("FEC:INFO:COLX[%d],ROWY[%d],I[%4d]: ", colx, rowy, i);
				m128i_print(val);
				printf("\n");
			}
		}
	}
}

// Row major 2D matrix
// Note2Self: X = Col, Y = Row
void fec_genfec(uint8_t *buf, int blocksize, int dmatrix)
{
	__m128i res, val;
	int iCurRowOffset;

	// Handle fec for each row of data blocks
	for(int rowy = 0; rowy < dmatrix; rowy++) {
		iCurRowOffset = rowy*(dmatrix+1)*blocksize;
		for(int i = 0; i < blocksize; i+= 16) {
			res = _mm_setzero_si128();
			for(int colx = 0; colx < dmatrix; colx++) {
				val = _mm_loadu_si128((__m128i*)&buf[iCurRowOffset+colx*blocksize+i]);
				res = _mm_xor_si128(res, val);
			}
			_mm_storeu_si128((__m128i*)&buf[iCurRowOffset+dmatrix*blocksize+i], res);
		}
	}
	// handle fec for each column of data blocks
	for(int colx = 0; colx < dmatrix; colx++) {
		for(int i = 0; i < blocksize; i+= 16) {
			res = _mm_setzero_si128();
			for(int rowy = 0; rowy < dmatrix; rowy++) {
				val = _mm_loadu_si128((__m128i*)&buf[rowy*(dmatrix+1)*blocksize+colx*blocksize+i]);
				res = _mm_xor_si128(res, val);
			}
			_mm_storeu_si128((__m128i*)&buf[dmatrix*(dmatrix+1)*blocksize+colx*blocksize+i], res);
		}
	}
}

void fec_checkfec(uint8_t *buf, int blocksize, int dmatrix, struct fecMatrixFlag *matFlag)
{
	__m128i res, val;
	int iCurRowOffset;
	__v4si v4i32;
	uint32_t iRes, *p32Res;

	// Handle fec for each row of data blocks
	for(int rowy = 0; rowy <= dmatrix; rowy++) {
		iCurRowOffset = rowy*(dmatrix+1)*blocksize;
		for(int i = 0; i < blocksize; i+= 16) {
			res = _mm_setzero_si128();
			for(int colx = 0; colx <= dmatrix; colx++) {
				val = _mm_loadu_si128((__m128i*)&buf[iCurRowOffset+colx*blocksize+i]);
				res = _mm_xor_si128(res, val);
			}
#ifdef FORCE_ERROR
			res = _mm_set_epi32(0x00,0x55,0xaa,0xff);
#endif
			v4i32 = (__v4si)res;
			iRes = v4i32[0] + v4i32[1] + v4i32[2] + v4i32[3];
			if (iRes != 0) {
				printf("FEC:WARN:ROWY=%d Not Valid\n", rowy);
				fecmatflag_rowset(matFlag, rowy);
			}
		}
	}
	// handle fec for each column of data blocks
	for(int colx = 0; colx <= dmatrix; colx++) {
		for(int i = 0; i < blocksize; i+= 16) {
			res = _mm_setzero_si128();
			for(int rowy = 0; rowy <= dmatrix; rowy++) {
				val = _mm_loadu_si128((__m128i*)&buf[rowy*(dmatrix+1)*blocksize+colx*blocksize+i]);
				res = _mm_xor_si128(res, val);
			}
			//iRes = res.m128i_i32[0] + res.m128i_i32[1];
			p32Res = (uint32_t*)&res;
			iRes = p32Res[0] + p32Res[1] + p32Res[2] + p32Res[3];
			if (iRes != 0) {
				printf("FEC:WARN:COLX=%d Not Valid\n", colx);
				fecmatflag_colset(matFlag, colx);
			}
		}
	}
}

void fec_recover(uint8_t *buf, int blocksize, int dmatrix, struct fecMatrixFlag *matFlag)
{
	int iNumErrBlocks, iErrCol, iErrRow;
	int iContinue, iDone;
	int iTryCnt;

	iTryCnt = 0;
	iDone = 1;
	iContinue = 1;
	while(iContinue) {
		printf("FEC:INFO:Recover: Take [%d] at recovery...\n", iTryCnt);
		iDone = 1;
		iContinue = 0;
		for (int i = 0; i < dmatrix; i++) {
			iNumErrBlocks = _mm_popcnt_u32(matFlag->rowview[i]);
			if (iNumErrBlocks == 0) {
				printf("FEC:INFO:GOOD:NO ERRBLOCKS: row[%d]\n", i);
			}
			if (iNumErrBlocks == 1) {
				iContinue = 1;
				printf("FEC:INFO:  OK:ONE ERRBLOCKS: row[%d]\n", i);
				iErrCol = __bsfd(matFlag->rowview[i]);
				printf("FEC:INFO: recovering err block at rowy[%d], colx[%d]\n", i, iErrCol);
				fecmatflag_blockclear(matFlag, i, iErrCol);
			}
			if (iNumErrBlocks > 1) {
				printf("FEC:INFO: BAD:MANY ERRBLOCKS: row[%d]\n", i);
				iDone = 0;
			}
		}
		for (int i = 0; i < dmatrix; i++) {
			iNumErrBlocks = _mm_popcnt_u32(matFlag->colview[i]);
			if (iNumErrBlocks == 0) {
				printf("FEC:INFO:GOOD:NO ERRBLOCKS: col[%d]\n", i);
			}
			if (iNumErrBlocks == 1) {
				iContinue = 1;
				printf("FEC:INFO:  OK:ONE ERRBLOCKS: col[%d]\n", i);
				iErrRow = __bsfd(matFlag->colview[i]);
				printf("FEC:INFO: recovering err block at rowy[%d], colx[%d]\n", iErrRow, i);
				fecmatflag_blockclear(matFlag, iErrRow, i);
			}
			if (iNumErrBlocks > 1) {
				printf("FEC:INFO: BAD:MANY ERRBLOCKS: col[%d]\n", i);
				iDone = 0;
			}
		}
		printf("FEC:INFO:Recover: iDone[%d], iContinue[%d]\n", iDone, iContinue);
		iTryCnt += 1;
	}
}

int fec_loadbuf(uint8_t *buf, int hFile, int blocksize, int dmatrix)
{
	int iRead;
	int iCurRowOffset;
	int iBufSizeDataRow = dmatrix*blocksize;
	for(int rowy = 0; rowy < dmatrix; rowy++) {
		iCurRowOffset = rowy*(dmatrix+1)*blocksize;
		iRead = read(hFile, &buf[iCurRowOffset], iBufSizeDataRow);
		if (iRead != iBufSizeDataRow)
			return -1;
	}
	return 0;
}

void fec_injecterror(uint8_t *buf, int blocksize, int dmatrix, struct fecMatrixFlag *matFlag)
{
	uint16_t iRand;
	int colx, rowy;

	for(int i = 0; i < 3; i++) {
		printf("FEC:INFO: injecting error [%d]:",i);
		if (_rdseed16_step(&iRand)) {
			colx = iRand % (dmatrix+1);
		} else {
			printf("RDSeed Failed\n");
		}
		if (_rdseed16_step(&iRand)) {
			rowy = iRand % (dmatrix+1);
		} else {
			printf("RDSeed Failed\n");
		}
		if (_rdseed16_step(&iRand)) {
			buf[rowy*(dmatrix+1)*blocksize+colx*blocksize+i] = iRand;
		} else {
			printf("RDSeed Failed\n");
		}
		fecmatflag_blockset(matFlag, rowy, colx);
		printf(" injected error at rowy[%d], colx[%d], i[%d] = 0x%X\n", rowy, colx, i, iRand);
	}
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
		fec_injecterror(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D, &gMatFlag);
		fec_checkfec(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D, &gMatFlag);
		fecmatflag_print(&gMatFlag, FEC_DATAMATRIX1D);
		fec_recover(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D, &gMatFlag);
		iMatrix += 1;
	}
	return 0;
}

