/*

Faster and better CRISPR guide RNA design with the Crackling method.
Jacob Bradford, Timothy Chappell, Dimitri Perrin
bioRxiv 2020.02.14.950261; doi: https://doi.org/10.1101/2020.02.14.950261


To compile:

nvcc -o isslScoreOfftargets isslScoreOfftargets.cu -Iinclude/ -O3 -std=c++11 -Xcompiler -fopenmp -Xcompiler -mpopcnt -Xcompiler -Iparallel_hashmap

*/

#include "cfdPenalties.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <chrono>
#include <bitset>
#include <iostream>
#include <climits>
#include <stdio.h>
#include <cstring>
#include <omp.h>
#include <phmap.h>
#include <map>

using namespace std;

size_t seqLength, seqCount, sliceWidth, sliceCount, offtargetsCount, scoresCount;

uint8_t nucleotideIndex[256];
vector<char> signatureIndex(4);
enum ScoreMethod { unknown = 0, mit = 1, cfd = 2, mitAndCfd = 3, mitOrCfd = 4, avgMitCfd = 5 };
struct CalcMethod { 
    bool mit;
    bool cfd;
};

/// Returns the size (bytes) of the file at `path`
size_t getFileSize(const char *path)
{
    struct stat64 statBuf;
    stat64(path, &statBuf);
    return statBuf.st_size;
}

/**
 * Binary encode genetic string `ptr`
 *
 * For example, 
 *   ATCG becomes
 *   00 11 01 10  (buffer with leading zeroes to encode as 64-bit unsigned int)
 *
 * @param[in] ptr the string containing ATCG to binary encode
 */
uint64_t sequenceToSignature(const char *ptr)
{
    uint64_t signature = 0;
    for (size_t j = 0; j < seqLength; j++) {
        signature |= (uint64_t)(nucleotideIndex[*ptr]) << (j * 2);
        ptr++;
    }
    return signature;
}

/**
 * Binary encode genetic string `ptr`
 *
 * For example, 
 *   00 11 01 10 becomes (as 64-bit unsigned int)
 *    A  T  C  G  (without spaces)
 *
 * @param[in] signature the binary encoded genetic string
 */
string signatureToSequence(uint64_t signature)
{
    string sequence = string(seqLength, ' ');
    for (size_t j = 0; j < seqLength; j++) {
        sequence[j] = signatureIndex[(signature >> (j * 2)) & 0x3];
    }
    return sequence;
}

__global__
void sequenceToSignatureCUDA(size_t queryCount, size_t seqLength, uint8_t *queryDataSet, uint64_t *querySignatures)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    uint8_t nucleotideIndex[256];

    nucleotideIndex['A'] = 0;
    nucleotideIndex['C'] = 1;
    nucleotideIndex['G'] = 2;
    nucleotideIndex['T'] = 3;

    for (size_t i = index; i < queryCount; i += stride) {
        uint8_t *ptr = &queryDataSet[i * (seqLength + 1)]; // (seqLength + 1) == seqLineLength

        uint64_t signature = 0;
        for (size_t j = 0; j < seqLength; j++) {
            signature |= (uint64_t)(nucleotideIndex[*ptr]) << (j * 2);
            ptr++;
        }

        querySignatures[i] = signature;
    }

}

struct constScoringArgs {
            uint64_t *offtargets;
            uint64_t *offtargetTogglesTail;
            int maxDist;
            CalcMethod calcMethod;
            ScoreMethod scoreMethod;
            double *precalculatedScores;
            double *cfdPamPenalties;
            double *cfdPosPenalties;
            double *totScoreMit;
            double *totScoreCfd;
            int *numOffTargetSitesScored;
            bool *checkNextSlice;
};

struct variScoringArgs {
            uint64_t *sliceOffset;
            uint64_t signaturesInSlice;
            uint64_t searchSignature;
            double maximum_sum;
};

__global__ void scoringCUDA(struct constScoringArgs* c_args, struct variScoringArgs* v_args) {
    uint64_t *offtargets = c_args->offtargets;
    uint64_t *offtargetTogglesTail = c_args->offtargetTogglesTail;
    int &maxDist = c_args->maxDist;
    CalcMethod &calcMethod = c_args->calcMethod;
    ScoreMethod &scoreMethod = c_args->scoreMethod;
    // double *precalculatedScores = c_args->precalculatedScores;
    double *cfdPamPenalties = c_args->cfdPamPenalties;
    double *cfdPosPenalties = c_args->cfdPosPenalties;
    double &totScoreMit = *c_args->totScoreMit;
    double &totScoreCfd = *c_args->totScoreCfd;
    int &numOffTargetSitesScored = *c_args->numOffTargetSitesScored;
    bool &checkNextSlice = *c_args->checkNextSlice;

    uint64_t *sliceOffset = v_args->sliceOffset;
    uint64_t &signaturesInSlice = v_args->signaturesInSlice;
    uint64_t searchSignature = v_args->searchSignature;
    double &maximum_sum = v_args->maximum_sum;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (size_t j = index; j < signaturesInSlice; j += stride) {
        uint64_t signatureWithOccurrencesAndId = sliceOffset[j];
        uint64_t signatureId = signatureWithOccurrencesAndId & 0xFFFFFFFFull;
        uint32_t occurrences = (signatureWithOccurrencesAndId >> (32));

        /** Find the positions of mismatches 
         *
         *  Search signature (SS):    A  A  T  T    G  C  A  T
         *                           00 00 11 11   10 01 00 11
         *              
         *        Off-target (OT):    A  T  A  T    C  G  A  T
         *                           00 11 00 11   01 10 00 11
         *                           
         *                SS ^ OT:   00 00 11 11   10 01 00 11
         *                         ^ 00 11 00 11   01 10 00 11
         *                  (XORd) = 00 11 11 00   11 11 00 00
         *
         *        XORd & evenBits:   00 11 11 00   11 11 00 00
         *                         & 10 10 10 10   10 10 10 10
         *                   (eX)  = 00 10 10 00   10 10 00 00
         *
         *         XORd & oddBits:   00 11 11 00   11 11 00 00
         *                         & 01 01 01 01   01 01 01 01
         *                   (oX)  = 00 01 01 00   01 01 00 00
         *
         *         (eX >> 1) | oX:   00 01 01 00   01 01 00 00 (>>1)
         *                         | 00 01 01 00   01 01 00 00
         *            mismatches   = 00 01 01 00   01 01 00 00
         *
         *   popcount(mismatches):   4
         */

        uint64_t xoredSignatures = searchSignature ^ offtargets[signatureId];
        uint64_t evenBits = xoredSignatures & 0xAAAAAAAAAAAAAAAAull;
        uint64_t oddBits = xoredSignatures & 0x5555555555555555ull;
        uint64_t mismatches = (evenBits >> 1) | oddBits;
        int dist = __popcll(mismatches);
        
        if (dist >= 0 && dist <= maxDist) {
            /** Prevent assessing the same off-target for multiple slices */
            uint64_t seenOfftargetAlready = 0;
            uint64_t * ptrOfftargetFlag = (offtargetTogglesTail - (signatureId / 64));
            seenOfftargetAlready = (*ptrOfftargetFlag >> (signatureId % 64)) & 1ULL;


            if (!seenOfftargetAlready) {
                // Begin calculating MIT score
                if (calcMethod.mit) {
                    if (dist > 0) {
                        // totScoreMit += precalculatedScores[mismatches] * (double)occurrences;
                    }
                } 
                
                // Begin calculating CFD score
                if (calcMethod.cfd) {
                    /** "In other words, for the CFD score, a value of 0 
                     *      indicates no predicted off-target activity whereas 
                     *      a value of 1 indicates a perfect match"
                     *      John Doench, 2016. 
                     *      https://www.nature.com/articles/nbt.3437
                    */
                    double cfdScore = 0;
                    if (dist == 0) {
                        cfdScore = 1;
                    }
                    else if (dist > 0 && dist <= maxDist) {
                        cfdScore = cfdPamPenalties[0b1010]; // PAM: NGG, TODO: do not hard-code the PAM
                        
                        for (size_t pos = 0; pos < 20; pos++) {
                            size_t mask = pos << 4;
                            
                            /** Create the mask to look up the position-identity score
                             *      In Python... c2b is char to bit
                             *       mask = pos << 4
                             *       mask |= c2b[sgRNA[pos]] << 2
                             *       mask |= c2b[revcom(offTaret[pos])]
                             *      
                             *      Find identity at `pos` for search signature
                             *      example: find identity in pos=2
                             *       Recall ISSL is inverted, hence:
                             *                   3'-  T  G  C  C  G  A -5'
                             *       start           11 10 01 01 10 00   
                             *       3UL << pos*2    00 00 00 11 00 00 
                             *       and             00 00 00 01 00 00
                             *       shift           00 00 00 00 01 00
                             */
                            uint64_t searchSigIdentityPos = searchSignature;
                            searchSigIdentityPos &= (3UL << (pos * 2));
                            searchSigIdentityPos = searchSigIdentityPos >> (pos * 2); 
                            searchSigIdentityPos = searchSigIdentityPos << 2;

                            /** Find identity at `pos` for offtarget
                             *      Example: find identity in pos=2
                             *      Recall ISSL is inverted, hence:
                             *                  3'-  T  G  C  C  G  A -5'
                             *      start           11 10 01 01 10 00   
                             *      3UL<<pos*2      00 00 00 11 00 00 
                             *      and             00 00 00 01 00 00
                             *      shift           00 00 00 00 00 01
                             *      rev comp 3UL    00 00 00 00 00 10 (done below)
                             */
                            uint64_t offtargetIdentityPos = offtargets[signatureId];
                            offtargetIdentityPos &= (3UL << (pos * 2));
                            offtargetIdentityPos = offtargetIdentityPos >> (pos * 2); 

                            /** Complete the mask
                             *      reverse complement (^3UL) `offtargetIdentityPos` here
                             */
                            mask = (mask | searchSigIdentityPos | (offtargetIdentityPos ^ 3UL));

                            if (searchSigIdentityPos >> 2 != offtargetIdentityPos) {
                                cfdScore *= cfdPosPenalties[mask];
                            }
                        }
                    }
                    atomicAdd(&totScoreCfd, cfdScore * (double)occurrences);
                }

                {
                    // atomicOr doesn't support 64-bit, so we split it into two 32-bit halves
                    // I don't know how expensive atomicOr is, so I've 'optimised' the second one out
                    uint32_t bit = signatureId % 64;
                    uint32_t is_high = bit > 31;
                    uint32_t flag32 = 1ULL << (bit - (is_high * 32));
                    atomicOr((uint32_t*)ptrOfftargetFlag + is_high, flag32);
                }

                atomicAdd(&numOffTargetSitesScored, occurrences);

                /** Stop calculating global score early if possible */
                switch (scoreMethod) {
                case ScoreMethod::mitAndCfd:
                    if (totScoreMit > maximum_sum &&
                        totScoreCfd > maximum_sum) {
                        checkNextSlice = false;
                    }
                    break;
                case ScoreMethod::mitOrCfd:
                    if (totScoreMit > maximum_sum ||
                        totScoreCfd > maximum_sum) {
                        checkNextSlice = false;
                    }
                    break;
                case ScoreMethod::avgMitCfd:
                    if (((totScoreMit + totScoreCfd) / 2.0) > maximum_sum) {
                        checkNextSlice = false;
                    }
                    break;
                case ScoreMethod::mit:
                    if (totScoreMit > maximum_sum) {
                        checkNextSlice = false;
                    }
                    break;
                case ScoreMethod::cfd:
                    if (totScoreCfd > maximum_sum) {
                        checkNextSlice = false;
                    }
                    break;
                default:
                    break;
                }
                if (!checkNextSlice) {
                    break;
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s [issltable] [query file] [max distance] [score-threshold] [score-method]\n", argv[0]);
        exit(1);
    }
    
    /** Char to binary encoding */
    nucleotideIndex['A'] = 0;
    nucleotideIndex['C'] = 1;
    nucleotideIndex['G'] = 2;
    nucleotideIndex['T'] = 3;
    signatureIndex[0] = 'A';
    signatureIndex[1] = 'C';
    signatureIndex[2] = 'G';
    signatureIndex[3] = 'T';

    /** The maximum number of mismatches */
    int maxDist = atoi(argv[3]);
    
    /** The threshold used to exit scoring early */
    double threshold = atof(argv[4]);
    
    /** Scoring methods. To exit early: 
     *      - only CFD must drop below `threshold`
     *      - only MIT must drop below `threshold`
     *      - both CFD and MIT must drop below `threshold`
     *      - CFD or MIT must drop below `threshold`
     *      - the average of CFD and MIT must below `threshold`
     */
	string argScoreMethod = argv[5];
    ScoreMethod scoreMethod = ScoreMethod::unknown;
    CalcMethod calcMethod = {false, false};
    if (!argScoreMethod.compare("and")) {
		scoreMethod = ScoreMethod::mitAndCfd;
        calcMethod.cfd = true;
        calcMethod.mit = true;
	} else if (!argScoreMethod.compare("or")) {
		scoreMethod = ScoreMethod::mitOrCfd;
        calcMethod.cfd = true;
        calcMethod.mit = true;
	} else if (!argScoreMethod.compare("avg")) {
		scoreMethod = ScoreMethod::avgMitCfd;
        calcMethod.cfd = true;
        calcMethod.mit = true;
	} else if (!argScoreMethod.compare("mit")) {
		scoreMethod = ScoreMethod::mit;
        calcMethod.mit = true;
	} else if (!argScoreMethod.compare("cfd")) {
		scoreMethod = ScoreMethod::cfd;
        calcMethod.cfd = true;
	}
	
    /** Begin reading the binary encoded ISSL, structured as:
     *      - a header (6 items)
     *      - precalcuated local MIT scores
     *      - all binary-encoded off-target sites
     *      - slice list sizes
     *      - slice contents
     */
    FILE *fp = fopen(argv[1], "rb");
    
    /** The index contains a fixed-sized header 
     *      - the number of off-targets in the index
     *      - the length of an off-target
     *      - 
     *      - chars per slice
     *      - the number of slices per sequence
     *      - the number of precalculated MIT scores
     */
    vector<size_t> slicelistHeader(6);
    
    if (fread(slicelistHeader.data(), sizeof(size_t), slicelistHeader.size(), fp) == 0) {
        fprintf(stderr, "Error reading index: header invalid\n");
        return 1;
    }
    
    offtargetsCount = slicelistHeader[0]; 
    seqLength       = slicelistHeader[1]; 
    seqCount        = slicelistHeader[2]; 
    sliceWidth      = slicelistHeader[3]; 
    sliceCount      = slicelistHeader[4]; 
    scoresCount     = slicelistHeader[5]; 
    
    /** The maximum number of possibly slice identities
     *      4 chars per slice * each of A,T,C,G = limit of 16
     */
    size_t sliceLimit = 1 << sliceWidth;
    
    /** Read in the precalculated MIT scores 
     *      - `mask` is a 2-bit encoding of mismatch positions
     *          For example,
     *              00 01 01 00 01  indicates mismatches in positions 1, 3 and 4
     *  
     *      - `score` is the local MIT score for this mismatch combination
     */
    phmap::flat_hash_map<uint64_t, double> precalculatedScores;

    for (int i = 0; i < scoresCount; i++) {
        uint64_t mask = 0;
        double score = 0.0;
        fread(&mask, sizeof(uint64_t), 1, fp);
        fread(&score, sizeof(double), 1, fp);
        
        precalculatedScores.insert(pair<uint64_t, double>(mask, score));
    }
    
    /** Load in all of the off-target sites */
    uint64_t* offtargets;
    // cudaMallocManaged(&offtargets, offtargetsCount * sizeof(uint64_t));
    // if (fread(offtargets, sizeof(uint64_t), offtargetsCount, fp) == 0) {
    //     fprintf(stderr, "Error reading index: loading off-target sequences failed\n");
    //     return 1;
    // }

    {
        size_t cuda_mem_free, cuda_mem_total;
        cudaMemGetInfo(&cuda_mem_free, &cuda_mem_total);

        printf("Free VRAM: %zu\n", cuda_mem_free);

        printf("Required VRAM: %zu\n", offtargetsCount);

        bool offtargetsPinned = offtargetsCount > cuda_mem_free;
        if (offtargetsPinned) {
            // Not enough VRAM
            fprintf(stderr, "Not enough VRAM, using CUDA managed memory\n");
            cudaMallocManaged(&offtargets, offtargetsCount * sizeof(uint64_t));
            if (fread(offtargets, sizeof(uint64_t), offtargetsCount, fp) == 0) {
                fprintf(stderr, "Error reading index: loading off-target sequences failed\n");
                return 1;
            }
        } else {
            // Enough VRAM, use device memory
            printf("Using VRAM\n");
            cudaMalloc(&offtargets, offtargetsCount * sizeof(uint64_t));

            vector<uint64_t> offtargetsTemp(offtargetsCount);
            if (fread(offtargetsTemp.data(), sizeof(uint64_t), offtargetsCount, fp) == 0) {
                fprintf(stderr, "Error reading index: loading off-target sequences failed\n");
                return 1;
            }

            cudaMemcpy(offtargets, offtargetsTemp.data(), offtargetsCount * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }
    }

    /** Prevent assessing an off-target site for multiple slices
     *
     *      Create enough 1-bit "seen" flags for the off-targets
     *      We only want to score a candidate guide against an off-target once.
     *      The least-significant bit represents the first off-target
     *      0 0 0 1   0 1 0 0   would indicate that the 3rd and 5th off-target have been seen.
     *      The CHAR_BIT macro tells us how many bits are in a byte (C++ >= 8 bits per byte)
     */
    uint64_t numOfftargetToggles = (offtargetsCount / ((size_t)sizeof(uint64_t) * (size_t)CHAR_BIT)) + 1;

    /** The number of signatures embedded per slice
     *
     *      These counts are stored contiguously
     *
     */
    vector<size_t> allSlicelistSizes(sliceCount * sliceLimit);
    
    if (fread(allSlicelistSizes.data(), sizeof(size_t), allSlicelistSizes.size(), fp) == 0) {
        fprintf(stderr, "Error reading index: reading slice list sizes failed\n");
        return 1;
    }
    
    /** The contents of the slices
     *
     *      Stored contiguously
     *
     *      Each signature (64-bit) is structured as:
     *          <occurrences 32-bit><off-target-id 32-bit>
     */
    // vector<uint64_t> allSignatures(seqCount * sliceCount);
    uint64_t *allSignatures;
    bool signaturesPinned;
    {
        size_t cuda_mem_free, cuda_mem_total;
        cudaMemGetInfo(&cuda_mem_free, &cuda_mem_total);

        printf("Free VRAM: %zu\n", cuda_mem_free);

        size_t allSignaturesSize = seqCount * sliceCount * sizeof(uint64_t);
        printf("Required VRAM: %zu\n", allSignaturesSize);

        signaturesPinned = allSignaturesSize > cuda_mem_free;
        if (signaturesPinned) {
            // Not enough VRAM
            // TODO: Use pinned memory
            fprintf(stderr, "Not enough VRAM, using CUDA managed memory\n");
            cudaMallocManaged(&allSignatures, allSignaturesSize);
            if (fread(allSignatures, sizeof(uint64_t), allSignaturesSize, fp) ==
                0) {
                fprintf(stderr,
                        "Error reading index: reading slice contents failed\n");
                return 1;
            }
        } else {
            // Enough VRAM, use device memory
            printf("Using VRAM\n");
            cudaMalloc(&allSignatures, allSignaturesSize);

            vector<uint64_t> allSignaturesTemp(seqCount * sliceCount);
            if (fread(allSignaturesTemp.data(), sizeof(uint64_t),
                      allSignaturesSize, fp) == 0) {
                fprintf(stderr,
                        "Error reading index: reading slice contents failed\n");
                return 1;
            }
            cudaMemcpy(allSignatures, allSignaturesTemp.data(),
                       allSignaturesSize, cudaMemcpyHostToDevice);

            // const size_t chunk_size = (8*1024*1024) / sizeof(uint64_t);
            // vector<uint64_t> allSignaturesTemp(chunk_size);
            // size_t left_to_read = allSignaturesSize-1;
            // uint64_t *ptr = allSignaturesTemp.data();
            // while (left_to_read > 0) {
            //     size_t to_read = min(left_to_read, chunk_size * sizeof(uint64_t));
            //     printf("Reading %zu\n", to_read);
            //     if (fread(ptr, sizeof(uint64_t), to_read, fp) == 0) {
            //         fprintf(stderr, "Error reading index: reading slice contents failed\n"); return 1;
            //     }
            //     cudaMemcpy(allSignatures, ptr, to_read,
            //     cudaMemcpyHostToDevice); left_to_read -= to_read;
            //     printf("Read %zu/%zu\n", allSignaturesSize - left_to_read,
            //     allSignaturesSize);
            // }
        }
    }

    /** End reading the index */
    fclose(fp);
    
    /** Start constructing index in memory
     *
     *      To begin, reverse the contiguous storage of the slices,
     *         into the following:
     *
     *         + Slice 0 :
     *         |---- AAAA : <slice contents>
     *         |---- AAAC : <slice contents>
     *         |----  ...
     *         | 
     *         + Slice 1 :
     *         |---- AAAA : <slice contents>
     *         |---- AAAC : <slice contents>
     *         |---- ...
     *         | ...
     */
    vector<vector<uint64_t *>> sliceLists(sliceCount, vector<uint64_t *>(sliceLimit));

    uint64_t *offset = allSignatures;
    for (size_t i = 0; i < sliceCount; i++) {
        for (size_t j = 0; j < sliceLimit; j++) {
            size_t idx = i * sliceLimit + j;
            sliceLists[i][j] = offset;
            offset += allSlicelistSizes[idx];
        }
    }
    
    /** Load query file (candidate guides)
     *      and prepare memory for calculated global scores
     */
    size_t seqLineLength = seqLength + 1;
    size_t fileSize = getFileSize(argv[2]);
    if (fileSize % seqLineLength != 0) {
        fprintf(stderr, "Error: query file is not a multiple of the expected line length (%zu)\n", seqLineLength);
        fprintf(stderr, "The sequence length may be incorrect; alternatively, the line endings\n");
        fprintf(stderr, "may be something other than LF, or there may be junk at the end of the file.\n");
        exit(1);
    }
    size_t queryCount = fileSize / seqLineLength;
    fp = fopen(argv[2], "rb");
    vector<char> queryDataSet(fileSize);
    vector<uint64_t> querySignatures(queryCount);
    vector<double> querySignatureMitScores(queryCount);
    vector<double> querySignatureCfdScores(queryCount);

    if (fread(queryDataSet.data(), fileSize, 1, fp) < 1) {
        fprintf(stderr, "Failed to read in query file.\n");
        exit(1);
    }
    fclose(fp);

    /** Binary encode query sequences */
#ifndef CPU_ONLY
    printf("Using CUDA\n");
    int blockSize = 1024; // This is a typical choice. It could be 128, 512, or 1024 depending on the GPU.
    int gridSize = (seqLength + blockSize - 1) / blockSize; // N is the size of your data

    uint8_t *d_queryDataSet;
    uint64_t *d_querySignatures;

    // Allocate VRAM for the query data
    cudaMalloc(&d_queryDataSet, queryCount * (seqLength + 1) * sizeof(uint8_t));
    cudaMalloc(&d_querySignatures, queryCount * sizeof(uint64_t));

    // Copy the query data to the VRAM
    cudaMemcpy(d_queryDataSet, &queryDataSet[0], queryCount * (seqLength + 1) * sizeof(uint8_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_querySignatures, &querySignatures[0], queryCount * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Calculate the signatures
    sequenceToSignatureCUDA<<<blockSize, gridSize>>>(queryCount, seqLength, d_queryDataSet, d_querySignatures);
    cudaDeviceSynchronize();
    cudaFree(d_queryDataSet);

    cudaMemcpy(&querySignatures[0], d_querySignatures, queryCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_querySignatures);


    double *d_cfdPamPenalties;
    double *d_cfdPosPenalties;
    cudaMalloc(&d_cfdPamPenalties, sizeof(cfdPamPenalties));
    cudaMalloc(&d_cfdPosPenalties, sizeof(cfdPosPenalties));
    cudaMemcpy(d_cfdPamPenalties, cfdPamPenalties, sizeof(cfdPamPenalties), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cfdPosPenalties, cfdPosPenalties, sizeof(cfdPosPenalties), cudaMemcpyHostToDevice);

    uint64_t *d_offtargetToggles;
    cudaMalloc(&d_offtargetToggles, numOfftargetToggles * sizeof(uint64_t));
    cudaMemset(d_offtargetToggles, 0, numOfftargetToggles * sizeof(uint64_t));
    uint64_t *d_offtargetTogglesTail = d_offtargetToggles + numOfftargetToggles - 1;

    double *d_totScoreMit;
    double *d_totScoreCfd;
    cudaMalloc(&d_totScoreMit, sizeof(double));
    cudaMalloc(&d_totScoreCfd, sizeof(double));

    int *d_numOffTargetSitesScored;
    cudaMalloc(&d_numOffTargetSitesScored, sizeof(int));

    bool *h_checkNextSlice, *d_checkNextSlice;
    cudaMallocHost(&h_checkNextSlice, sizeof(bool));  // Use pinned memory for faster transfers
    cudaMalloc(&d_checkNextSlice, sizeof(bool));

    // 256 byte limit on the number of arguments to a kernel, so we use a struct
    struct constScoringArgs scoring_args = {
        .offtargets = offtargets,
        .offtargetTogglesTail = d_offtargetTogglesTail,
        .maxDist = maxDist,
        .calcMethod = calcMethod,
        .scoreMethod = scoreMethod,
        .precalculatedScores = NULL,  // TODO: precalculatedScores
        .cfdPamPenalties = d_cfdPamPenalties,
        .cfdPosPenalties = d_cfdPosPenalties,
        .totScoreMit = d_totScoreMit,
        .totScoreCfd = d_totScoreCfd,
        .numOffTargetSitesScored = d_numOffTargetSitesScored,
        .checkNextSlice = d_checkNextSlice,
    };

    struct constScoringArgs* d_scoring_args;
    cudaMalloc(&d_scoring_args, sizeof(struct constScoringArgs));
    cudaMemcpy(d_scoring_args, &scoring_args, sizeof(struct constScoringArgs), cudaMemcpyHostToDevice);

    struct variScoringArgs slice_args = {
        .sliceOffset = NULL,
        .signaturesInSlice = 0,
        .searchSignature = 0,
        .maximum_sum = 0.0,
    };

    struct variScoringArgs* d_slice_args;
    cudaMalloc(&d_slice_args, sizeof(struct variScoringArgs));

    unordered_map<uint64_t, unordered_set<uint64_t>> searchResults;
    // vector<uint64_t> offtargetToggles(numOfftargetToggles);

    // uint64_t * offtargetTogglesTail = offtargetToggles.data() + numOfftargetToggles - 1;

    /** For each candidate guide */
    // TODO: Remove starting offset before finishing
    for (size_t searchIdx = querySignatures.size() - 16; searchIdx < querySignatures.size(); searchIdx++) {

        auto searchSignature = querySignatures[searchIdx];

        /** Global scores */
        cudaMemset(d_totScoreMit, 0, sizeof(double));  // double totScoreMit = 0.0;
        cudaMemset(d_totScoreCfd, 0, sizeof(double));  // double totScoreCfd = 0.0;
        
        cudaMemset(d_numOffTargetSitesScored, 0, sizeof(int));  // int numOffTargetSitesScored = 0;
        double maximum_sum = (10000.0 - threshold*100) / threshold;

        *h_checkNextSlice = true;
        cudaMemcpy(d_checkNextSlice, h_checkNextSlice, sizeof(bool), cudaMemcpyHostToDevice);

        /** For each ISSL slice */
        for (size_t i = 0; i < sliceCount; i++) {
            uint64_t sliceMask = sliceLimit - 1;
            int sliceShift = sliceWidth * i;
            sliceMask = sliceMask << sliceShift;
            
            uint64_t searchSlice = (searchSignature & sliceMask) >> sliceShift;
            
            size_t idx = i * sliceLimit + searchSlice;
            
            size_t signaturesInSlice = allSlicelistSizes[idx];
            uint64_t *sliceOffset = sliceLists[i][searchSlice];

            /** For each off-target signature in slice */

            slice_args = {
                .sliceOffset = sliceOffset,
                .signaturesInSlice = signaturesInSlice,
                .searchSignature = searchSignature,
                .maximum_sum = maximum_sum,
            };

            // Check last run to see if we should continue
            cudaDeviceSynchronize();
            cudaMemcpy(h_checkNextSlice, d_checkNextSlice, sizeof(bool), cudaMemcpyDeviceToHost);
            if (!*h_checkNextSlice)
                break;

            cudaMemcpy(d_slice_args, &slice_args, sizeof(struct variScoringArgs), cudaMemcpyHostToDevice);

            scoringCUDA<<<blockSize, gridSize>>>(d_scoring_args, d_slice_args);
            // Continue next iteration to prepare the next run
        }
        cudaDeviceSynchronize();

        double totScoreMit = 0.0;
        double totScoreCfd = 0.0;
        cudaMemcpy(&totScoreMit, d_totScoreMit, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&totScoreCfd, d_totScoreCfd, sizeof(double), cudaMemcpyDeviceToHost);

        querySignatureMitScores[searchIdx] = 10000.0 / (100.0 + totScoreMit);
        querySignatureCfdScores[searchIdx] = 10000.0 / (100.0 + totScoreCfd);

        // memset(offtargetToggles.data(), 0, sizeof(uint64_t)*offtargetToggles.size());
        cudaMemset(d_offtargetToggles, 0, numOfftargetToggles * sizeof(uint64_t));

        // printf("Processed %zu/%zu\n", searchIdx, querySignatures.size());
    }

    // Free VRAM
    cudaFree(d_cfdPamPenalties);
    cudaFree(d_cfdPosPenalties);
    cudaFree(d_offtargetToggles);
    cudaFree(d_totScoreMit);
    cudaFree(d_totScoreCfd);
    cudaFree(d_numOffTargetSitesScored);
    cudaFree(d_scoring_args);
    cudaFree(d_checkNextSlice);
    cudaFreeHost(h_checkNextSlice);

#else
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t i = 0; i < queryCount; i++) {
            char *ptr = &queryDataSet[i * seqLineLength];
            uint64_t signature = sequenceToSignature(ptr);
            querySignatures[i] = signature;
        }
    }

    /** Begin scoring */
    #pragma omp parallel
    {
        unordered_map<uint64_t, unordered_set<uint64_t>> searchResults;
        vector<uint64_t> offtargetToggles(numOfftargetToggles);
    
        uint64_t * offtargetTogglesTail = offtargetToggles.data() + numOfftargetToggles - 1;

        /** For each candidate guide */
        #pragma omp for
        // TODO: Remove starting offset before finishing
        for (size_t searchIdx = querySignatures.size() - 32; searchIdx < querySignatures.size(); searchIdx++) {

            auto searchSignature = querySignatures[searchIdx];

            /** Global scores */
            double totScoreMit = 0.0;
            double totScoreCfd = 0.0;
            
            int numOffTargetSitesScored = 0;
            double maximum_sum = (10000.0 - threshold*100) / threshold;
            bool checkNextSlice = true;
            
            /** For each ISSL slice */
            for (size_t i = 0; i < sliceCount; i++) {
                uint64_t sliceMask = sliceLimit - 1;
                int sliceShift = sliceWidth * i;
                sliceMask = sliceMask << sliceShift;
                auto &sliceList = sliceLists[i];
                
                uint64_t searchSlice = (searchSignature & sliceMask) >> sliceShift;
                
                size_t idx = i * sliceLimit + searchSlice;
                
                size_t signaturesInSlice = allSlicelistSizes[idx];
                uint64_t *sliceOffset = sliceList[searchSlice];
                
                /** For each off-target signature in slice */
                for (size_t j = 0; j < signaturesInSlice; j++) {
                    
                    auto signatureWithOccurrencesAndId = sliceOffset[j];
                    auto signatureId = signatureWithOccurrencesAndId & 0xFFFFFFFFull;
                    uint32_t occurrences = (signatureWithOccurrencesAndId >> (32));

                    /** Find the positions of mismatches 
                     *
                     *  Search signature (SS):    A  A  T  T    G  C  A  T
                     *                           00 00 11 11   10 01 00 11
                     *              
                     *        Off-target (OT):    A  T  A  T    C  G  A  T
                     *                           00 11 00 11   01 10 00 11
                     *                           
                     *                SS ^ OT:   00 00 11 11   10 01 00 11
                     *                         ^ 00 11 00 11   01 10 00 11
                     *                  (XORd) = 00 11 11 00   11 11 00 00
                     *
                     *        XORd & evenBits:   00 11 11 00   11 11 00 00
                     *                         & 10 10 10 10   10 10 10 10
                     *                   (eX)  = 00 10 10 00   10 10 00 00
                     *
                     *         XORd & oddBits:   00 11 11 00   11 11 00 00
                     *                         & 01 01 01 01   01 01 01 01
                     *                   (oX)  = 00 01 01 00   01 01 00 00
                     *
                     *         (eX >> 1) | oX:   00 01 01 00   01 01 00 00 (>>1)
                     *                         | 00 01 01 00   01 01 00 00
                     *            mismatches   = 00 01 01 00   01 01 00 00
                     *
                     *   popcount(mismatches):   4
                     */
                    uint64_t xoredSignatures = searchSignature ^ offtargets[signatureId];
                    uint64_t evenBits = xoredSignatures & 0xAAAAAAAAAAAAAAAAull;
                    uint64_t oddBits = xoredSignatures & 0x5555555555555555ull;
                    uint64_t mismatches = (evenBits >> 1) | oddBits;
                    int dist = __builtin_popcountll(mismatches);
					
					if (dist >= 0 && dist <= maxDist) {

						/** Prevent assessing the same off-target for multiple slices */
						uint64_t seenOfftargetAlready = 0;
						uint64_t * ptrOfftargetFlag = (offtargetTogglesTail - (signatureId / 64));
						seenOfftargetAlready = (*ptrOfftargetFlag >> (signatureId % 64)) & 1ULL;
                    
										
						if (!seenOfftargetAlready) {
							// Begin calculating MIT score
							if (calcMethod.mit) {
								if (dist > 0) {
									// totScoreMit += precalculatedScores[mismatches] * (double)occurrences;
								}
							} 
							
							// Begin calculating CFD score
							if (calcMethod.cfd) {
								/** "In other words, for the CFD score, a value of 0 
								 *      indicates no predicted off-target activity whereas 
								 *      a value of 1 indicates a perfect match"
								 *      John Doench, 2016. 
								 *      https://www.nature.com/articles/nbt.3437
								*/
								double cfdScore = 0;
								if (dist == 0) {
									cfdScore = 1;
								}
								else if (dist > 0 && dist <= maxDist) {
									cfdScore = cfdPamPenalties[0b1010]; // PAM: NGG, TODO: do not hard-code the PAM
									
									for (size_t pos = 0; pos < 20; pos++) {
										size_t mask = pos << 4;
										
										/** Create the mask to look up the position-identity score
										 *      In Python... c2b is char to bit
										 *       mask = pos << 4
										 *       mask |= c2b[sgRNA[pos]] << 2
										 *       mask |= c2b[revcom(offTaret[pos])]
										 *      
										 *      Find identity at `pos` for search signature
										 *      example: find identity in pos=2
										 *       Recall ISSL is inverted, hence:
										 *                   3'-  T  G  C  C  G  A -5'
										 *       start           11 10 01 01 10 00   
										 *       3UL << pos*2    00 00 00 11 00 00 
										 *       and             00 00 00 01 00 00
										 *       shift           00 00 00 00 01 00
										 */
										uint64_t searchSigIdentityPos = searchSignature;
										searchSigIdentityPos &= (3UL << (pos * 2));
										searchSigIdentityPos = searchSigIdentityPos >> (pos * 2); 
										searchSigIdentityPos = searchSigIdentityPos << 2;

										/** Find identity at `pos` for offtarget
										 *      Example: find identity in pos=2
										 *      Recall ISSL is inverted, hence:
										 *                  3'-  T  G  C  C  G  A -5'
										 *      start           11 10 01 01 10 00   
										 *      3UL<<pos*2      00 00 00 11 00 00 
										 *      and             00 00 00 01 00 00
										 *      shift           00 00 00 00 00 01
										 *      rev comp 3UL    00 00 00 00 00 10 (done below)
										 */
										uint64_t offtargetIdentityPos = offtargets[signatureId];
										offtargetIdentityPos &= (3UL << (pos * 2));
										offtargetIdentityPos = offtargetIdentityPos >> (pos * 2); 

										/** Complete the mask
										 *      reverse complement (^3UL) `offtargetIdentityPos` here
										 */
										mask = (mask | searchSigIdentityPos | (offtargetIdentityPos ^ 3UL));

										if (searchSigIdentityPos >> 2 != offtargetIdentityPos) {
											cfdScore *= cfdPosPenalties[mask];
										}
									}
								}
								totScoreCfd += cfdScore * (double)occurrences;
							}
					
							*ptrOfftargetFlag |= (1ULL << (signatureId % 64));
							numOffTargetSitesScored += occurrences;

							/** Stop calculating global score early if possible */
							if (scoreMethod == ScoreMethod::mitAndCfd) {
								if (totScoreMit > maximum_sum && totScoreCfd > maximum_sum) {
									checkNextSlice = false;
									break;
								}
							}
							if (scoreMethod == ScoreMethod::mitOrCfd) {
								if (totScoreMit > maximum_sum || totScoreCfd > maximum_sum) {
									checkNextSlice = false;
									break;
								}
							}
							if (scoreMethod == ScoreMethod::avgMitCfd) {
								if (((totScoreMit + totScoreCfd) / 2.0) > maximum_sum) {
									checkNextSlice = false;
									break;
								}
							}
							if (scoreMethod == ScoreMethod::mit) {
								if (totScoreMit > maximum_sum) {
									checkNextSlice = false;
									break;
								}
							}
							if (scoreMethod == ScoreMethod::cfd) {
								if (totScoreCfd > maximum_sum) {
									checkNextSlice = false;
									break;
								}
							}
						}
					}
                }

                if (!checkNextSlice)
                    break;
            }

            querySignatureMitScores[searchIdx] = 10000.0 / (100.0 + totScoreMit);
            querySignatureCfdScores[searchIdx] = 10000.0 / (100.0 + totScoreCfd);

            memset(offtargetToggles.data(), 0, sizeof(uint64_t)*offtargetToggles.size());
        }

    }
#endif
    /** Print global scores to stdout */
    // TODO: Remove starting offset before finishing
    for (size_t searchIdx = querySignatures.size() - 128; searchIdx < querySignatures.size(); searchIdx++) {
        auto querySequence = signatureToSequence(querySignatures[searchIdx]);
        printf("%s\t", querySequence.c_str());
        if (calcMethod.mit) 
            printf("%f\t", querySignatureMitScores[searchIdx]);
        else
            printf("-1\t");
        
        if (calcMethod.cfd)
            printf("%f\n", querySignatureCfdScores[searchIdx]);
        else
            printf("-1\n");
            
    }

    return 0;
}