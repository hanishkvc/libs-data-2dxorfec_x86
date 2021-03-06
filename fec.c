/*
 * 2D XOR based FEC
 * v20160820_1940
 * HanishKVC, 2016
 *
 * This is a new reimplementation of my very old 2d xor based fec logic.
 * It uses SIMD and other special instructions supported by x86 to help
 * speed up the fec logic.
 *
 * I have avoided looking at my old logic/implementation, to allow new
 * ideas/logic to emerge if and where possible naturally. Have to check
 * with my old logic later to see, how similar or dissimilar is some of
 * the implementation details, in this new take on my old concept.
 *
 * Dedicated to my elders.
 */
#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h>
#include <x86intrin.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#ifndef FEC_TARGETHAS_RDSEED
#include <sys/time.h>
#endif

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

void fec_weakcrosscheck_amongflags(struct fecMatrixFlag *matFlag, int dmatrix)
{
	uint32_t uBitPos;
	uint32_t curRowSet, curColSet;
	uint32_t curRowSetCnt, curColSetCnt;
	int iErrorOrDebug = 0;

	for(int i = 0; i <= dmatrix; i++) {
		uBitPos = (1 << i);
		// Check along row
		curRowSet = matFlag->row & uBitPos;
		curRowSetCnt = _mm_popcnt_u32(matFlag->rowview[i]);
		if (curRowSet) {
			if (curRowSetCnt == 0) {
				iErrorOrDebug = 1;
				printf("FEC:ERROR:weakcrosscheck: checkfec says row[%d] has error, but rowview[%d] doesnt have any bit set\n", i, i);
			}
		}
		if (curRowSetCnt != 0) {
			if (curRowSet == 0) {
				iErrorOrDebug = 1;
				printf("FEC:DEBUG:weakcrosscheck: rowview[%d] has bits set, but checkfec didnt find error: Either cyclical complementing errors present, which cant be detected at global level OR bug in checkfec?\n", i);
			}
		}
		// Check along col
		curColSet = matFlag->col & uBitPos;
		curColSetCnt = _mm_popcnt_u32(matFlag->colview[i]);
		if (curColSet) {
			if (curColSetCnt == 0) {
				iErrorOrDebug = 1;
				printf("FEC:ERROR:weakcrosscheck: checkfec says col[%d] has error, but colview[%d] doesnt have any bit set\n", i, i);
			}
		}
		if (curColSetCnt != 0) {
			if (curColSet == 0) {
				iErrorOrDebug = 1;
				printf("FEC:DEBUG:weakcrosscheck: colview[%d] has bits set, but checkfec didnt find error: Either cyclical complementing errors present, which cant be detected at global level OR bug in checkfec?\n", i);
			}
		}
	}
	if (iErrorOrDebug == 0) {
		printf("FEC:GOOD:weakcrosscheck: Didn't find any issues, but remember this is a weak check, some complementing errors can slip through\n");
	} else {
		printf("FEC:GOOD:weakcrosscheck: If there is a FEC:DEBUG: msg above corresponding to fec row/col, in most cases it can be ignored as checkfec ignores fec row/col because fec row/col doesn't have a fec block of its own, for its own recovery from within\n");
		printf("FEC:GOOD:weakcrosscheck: Found issues, but remember this is a weak check, some complementing errors can confuse the logic\n");
	}
}

void fec_checkfec(uint8_t *buf, int blocksize, int dmatrix, struct fecMatrixFlag *matFlag)
{
	__m128i res, val;
	int iCurRowOffset;
	__v4si v4i32;
	uint32_t iRes, *p32Res;

	// Check fec for each row of data blocks
	for(int rowy = 0; rowy < dmatrix; rowy++) {
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
	// Check fec for each column of data blocks
	for(int colx = 0; colx < dmatrix; colx++) {
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

	// Cross check between error block flags set by user program and Global error identification above
	fec_weakcrosscheck_amongflags(matFlag, dmatrix);
}

#define FEC_RECOVER_ALONGROW 0
#define FEC_RECOVER_ALONGCOL 1

void fec_recoverblock(uint8_t *buf, int blocksize, int dmatrix, struct fecMatrixFlag *matFlag, int errBlockRowY, int errBlockColX, int along)
{
	__m128i res, val;
	int iCurRowOffset;
	if (along == FEC_RECOVER_ALONGROW) {
		iCurRowOffset = errBlockRowY*(dmatrix+1)*blocksize;
		for(int i = 0; i < blocksize; i+= 16) {
			res = _mm_setzero_si128();
			for(int colx = 0; colx <= dmatrix; colx++) {
				if (colx == errBlockColX) {
					continue;
				}
				val = _mm_loadu_si128((__m128i*)&buf[iCurRowOffset+colx*blocksize+i]);
				res = _mm_xor_si128(res, val);
			}
			_mm_storeu_si128((__m128i*)&buf[iCurRowOffset+errBlockColX*blocksize+i], res);
		}
	} else {		// FEC_RECOVER_ALONGCOL
		for(int i = 0; i < blocksize; i+= 16) {
			res = _mm_setzero_si128();
			for(int rowy = 0; rowy <= dmatrix; rowy++) {
				if (rowy == errBlockRowY) {
					continue;
				}
				val = _mm_loadu_si128((__m128i*)&buf[rowy*(dmatrix+1)*blocksize+errBlockColX*blocksize+i]);
				res = _mm_xor_si128(res, val);
			}
			_mm_storeu_si128((__m128i*)&buf[errBlockRowY*(dmatrix+1)*blocksize+errBlockColX*blocksize+i], res);
		}
	}
	fecmatflag_blockclear(matFlag, errBlockRowY, errBlockColX);
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
		// Check along the rows
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
				fec_recoverblock(buf, blocksize, dmatrix, matFlag, i, iErrCol, FEC_RECOVER_ALONGROW);
			}
			if (iNumErrBlocks > 1) {
				printf("FEC:INFO: BAD:MANY ERRBLOCKS: row[%d]\n", i);
				iDone = 0;
			}
		}
		// Check along the columns
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
				fec_recoverblock(buf, blocksize, dmatrix, matFlag, iErrRow, i, FEC_RECOVER_ALONGCOL);
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

#define FEC_BUFFILE_DATAONLY 0
#define FEC_BUFFILE_DATAFEC 1

int fec_loadbuf(uint8_t *buf, int hFile, int blocksize, int dmatrix, int mode)
{
	int iRead;
	int iCurRowOffset;
	int iBufSizeDataOnlyRow = dmatrix*blocksize;
	int iBufSizeDataFecRow = (dmatrix+1)*blocksize;
	int iBufSizeForRow = 0;
	int iRowCount = 0;

	if (mode == FEC_BUFFILE_DATAONLY) {
		iBufSizeForRow = iBufSizeDataOnlyRow;
		iRowCount = dmatrix;
	} else {
		iBufSizeForRow = iBufSizeDataFecRow;
		iRowCount = dmatrix+1;
	}

	for(int rowy = 0; rowy < iRowCount; rowy++) {
		iCurRowOffset = rowy*(dmatrix+1)*blocksize;
		iRead = read(hFile, &buf[iCurRowOffset], iBufSizeForRow);
		if (iRead != iBufSizeForRow)
			return -1;
	}
	return 0;
}

int fec_storebuf(uint8_t *buf, int hFile, int blocksize, int dmatrix, int mode)
{
	int iWrote;
	int iCurRowOffset;
	int iBufSizeDataOnlyRow = dmatrix*blocksize;
	int iBufSizeDataFecRow = (dmatrix+1)*blocksize;
	int iBufSizeForRow = 0;
	int iRowCount = 0;

	if (mode == FEC_BUFFILE_DATAONLY) {
		iBufSizeForRow = iBufSizeDataOnlyRow;
		iRowCount = dmatrix;
	} else {
		iBufSizeForRow = iBufSizeDataFecRow;
		iRowCount = dmatrix+1;
	}

	for(int rowy = 0; rowy < iRowCount; rowy++) {
		iCurRowOffset = rowy*(dmatrix+1)*blocksize;
		iWrote = write(hFile, &buf[iCurRowOffset], iBufSizeForRow);
		if (iWrote != iBufSizeForRow)
			return -1;
	}
	return 0;
}

int fec_getrandom16(uint16_t *rand)
{
	int res = 0;
#ifdef FEC_TARGETHAS_RDSEED
	res = _rdseed16_step(rand);
#else
	static int inited = 0;
	struct timeval tv;

	if (inited == 0) {
		gettimeofday(&tv, NULL);
		srandom(tv.tv_sec);
		inited = 1;
	}
	res = 1;
	*rand = random();
#endif
	return res;
}

void fec_injecterror(uint8_t *buf, int blocksize, int dmatrix, struct fecMatrixFlag *matFlag)
{
	uint16_t iRand;
	int colx, rowy;

	for(int i = 0; i < 3; i++) {
		printf("FEC:INFO: injecting error [%d]:",i);
		if (fec_getrandom16(&iRand)) {
			colx = iRand % (dmatrix+1);
		} else {
			printf("RDSeed Failed\n");
		}
		if (fec_getrandom16(&iRand)) {
			rowy = iRand % (dmatrix+1);
		} else {
			printf("RDSeed Failed\n");
		}
		if (fec_getrandom16(&iRand)) {
			buf[rowy*(dmatrix+1)*blocksize+colx*blocksize+i] = iRand;
		} else {
			printf("RDSeed Failed\n");
		}
		fecmatflag_blockset(matFlag, rowy, colx);
		printf(" injected error at rowy[%d], colx[%d], i[%d] = 0x%X\n", rowy, colx, i, iRand);
	}
}

#define FEC_PRGMODE_GENFEC 0
#define FEC_PRGMODE_USEFEC 1
#define FEC_PRGMODE_ALLFEC 2

int test_genfec(int hFSrc, int hFDst)
{
	uint8_t buf[FEC_BUFFERSIZE];
	int iMatrix;

	memset(buf, 0, FEC_BUFFERSIZE);
	iMatrix = 0;
	while(1) {
		printf("FEC:INFO: processing data matrix [%d]\n", iMatrix);
		if (fec_loadbuf(buf, hFSrc, FEC_BLOCKSIZE, FEC_DATAMATRIX1D, FEC_BUFFILE_DATAONLY) != 0) {
			printf("FEC:INFO: failed loading of data matrix [%d], maybe EOF, quiting...\n", iMatrix);
			return -1;
		} else {
			printf("FEC:INFO: loaded data matrix [%d]\n", iMatrix);
		}
		fec_printbuf_start(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D);
		fec_genfec(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D);
		fec_printbuf_start(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D);
		if (fec_storebuf(buf, hFDst, FEC_BLOCKSIZE, FEC_DATAMATRIX1D, FEC_BUFFILE_DATAFEC) != 0) {
			printf("FEC:INFO: failed storing of data+fec matrix [%d], maybe EOF, quiting...\n", iMatrix);
			return -1;
		} else {
			printf("FEC:INFO: stored data+fec matrix [%d]\n", iMatrix);
		}
		iMatrix += 1;
	}
	return 0;
}

int test_usefec(int hFSrc, int hFDst)
{
	uint8_t buf[FEC_BUFFERSIZE];
	int iMatrix;

	memset(buf, 0, FEC_BUFFERSIZE);
	iMatrix = 0;
	while(1) {
		printf("FEC:INFO: processing data+fec matrix [%d]\n", iMatrix);
		if (fec_loadbuf(buf, hFSrc, FEC_BLOCKSIZE, FEC_DATAMATRIX1D, FEC_BUFFILE_DATAFEC) != 0) {
			printf("FEC:INFO: failed loading of data+fec matrix [%d], maybe EOF, quiting...\n", iMatrix);
			return -1;
		} else {
			printf("FEC:INFO: loaded data+fec matrix [%d]\n", iMatrix);
		}
		fec_printbuf_start(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D);
		fec_injecterror(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D, &gMatFlag);
		fec_checkfec(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D, &gMatFlag);
		fecmatflag_print(&gMatFlag, FEC_DATAMATRIX1D);
		fec_recover(buf, FEC_BLOCKSIZE, FEC_DATAMATRIX1D, &gMatFlag);
		if (fec_storebuf(buf, hFDst, FEC_BLOCKSIZE, FEC_DATAMATRIX1D, FEC_BUFFILE_DATAONLY) != 0) {
			printf("FEC:INFO: failed storing of data matrix [%d], maybe EOF, quiting...\n", iMatrix);
			return -1;
		} else {
			printf("FEC:INFO: stored data matrix [%d]\n", iMatrix);
		}
		iMatrix += 1;
	}
	return 0;
}

int test_all(int hFSrc, int hFDst)
{
	uint8_t buf[FEC_BUFFERSIZE];
	int iMatrix;

	memset(buf, 0, FEC_BUFFERSIZE);
	iMatrix = 0;
	while(1) {
		printf("FEC:INFO: processing data matrix [%d]\n", iMatrix);
		if (fec_loadbuf(buf, hFSrc, FEC_BLOCKSIZE, FEC_DATAMATRIX1D, FEC_BUFFILE_DATAONLY) != 0) {
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

int main(int argc, char **argv)
{
	int hFSrc, hFDst;
	int iMode;

	if (strncmp(argv[1], "gen",3) == 0) {
		iMode = FEC_PRGMODE_GENFEC;
	} else {
		if (strncmp(argv[1], "use",3) == 0) {
			iMode = FEC_PRGMODE_USEFEC;
		} else {
			iMode = FEC_PRGMODE_ALLFEC;
		}
	}
	hFSrc = open(argv[2], O_RDONLY);
	hFDst = open(argv[3], O_CREAT | O_WRONLY, S_IRUSR);


	switch(iMode) {
		case FEC_PRGMODE_GENFEC:
			test_genfec(hFSrc, hFDst);
			break;
		case FEC_PRGMODE_USEFEC:
			test_usefec(hFSrc, hFDst);
			break;
		default:
			test_all(hFSrc, hFDst);
	}

	return 0;
}

