/*

Faster and better CRISPR guide RNA design with the Crackling method.
Jacob Bradford, Timothy Chappell, Dimitri Perrin
bioRxiv 2020.02.14.950261; doi: https://doi.org/10.1101/2020.02.14.950261

A CUDA 6.0+ compatible GPU is required for this CUDA version of Crackling (Pascal/GTX 10xx or newer).
Enough VRAM is basically essential (but it will run very slowly with less).

To compile:

nvcc -o isslScoreOfftargets isslScoreOfftargets.cu -O3 -Iinclude/ -arch=sm_60

Consider using native compilation for the best performance on your system:
  * Set `-arch=native` for native GPU compilation
  * Add `-Xcompiler -march=native` for native CPU compilation (has minimal effect)

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
#include <bitset>
#include <iostream>
// #include <omp.h>

using namespace std;

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
void sequenceToSignatureCUDA(uint64_t queryCount, uint64_t seqLength, uint8_t *queryDataSet, uint64_t *querySignatures)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    uint8_t nucleotideIndex[256];

    nucleotideIndex['A'] = 0;
    nucleotideIndex['C'] = 1;
    nucleotideIndex['G'] = 2;
    nucleotideIndex['T'] = 3;

    for (uint64_t i = index; i < queryCount; i += stride) {
        uint8_t *ptr = &queryDataSet[i * (seqLength + 1)]; // (seqLength + 1) == seqLineLength

        uint64_t signature = 0;
        for (uint32_t j = 0; j < seqLength; j++) {
            signature |= (uint64_t)(nucleotideIndex[*ptr]) << (j * 2);
            ptr++;
        }

        querySignatures[i] = signature;
    }

}

struct constScoringArgs {
            uint64_t *offtargets;
            uint64_t *offtargetTogglesTail;
            uint32_t maxDist;
            CalcMethod calcMethod;
            ScoreMethod scoreMethod;
            double *precalculatedScores;
            double *cfdPamPenalties;
            double *cfdPosPenalties;
            double *totScoreMit;
            double *totScoreCfd;
            uint32_t *numOffTargetSitesScored;
            bool *checkNextSlice;
};

// This should be faster to read on the GPU than checkNextSlice (which is on the CPU)
__device__ bool continueCUDA = true;

// Makes use of constant memory
__constant__ struct constScoringArgs d_constant_args;

__global__ void scoringCUDA(uint64_t *sliceOffset, const uint64_t signaturesInSlice, const uint64_t searchSignature, const double maximum_sum) {
    // Extract constScoringArgs into easier to use variables
    uint64_t *offtargets = d_constant_args.offtargets;
    uint64_t *offtargetTogglesTail = d_constant_args.offtargetTogglesTail;
    const uint32_t maxDist = d_constant_args.maxDist;
    const CalcMethod calcMethod = d_constant_args.calcMethod;
    const ScoreMethod scoreMethod = d_constant_args.scoreMethod;
    double *precalculatedScores = d_constant_args.precalculatedScores;
    double *cfdPamPenalties = d_constant_args.cfdPamPenalties;
    double *cfdPosPenalties = d_constant_args.cfdPosPenalties;
    double &totScoreMit = *d_constant_args.totScoreMit;
    double &totScoreCfd = *d_constant_args.totScoreCfd;
    uint32_t &numOffTargetSitesScored = *d_constant_args.numOffTargetSitesScored;
    bool &checkNextSlice = *d_constant_args.checkNextSlice;

    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    continueCUDA = true;

    /** For each off-target signature in slice */
    for (uint64_t j = index; j < signaturesInSlice; j += stride) {
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
        uint32_t dist = __popcll(mismatches);
        
        if (dist <= maxDist) {
            /** Prevent assessing the same off-target for multiple slices */
            uint64_t * ptrOfftargetFlag = (offtargetTogglesTail - (signatureId / 64));
            uint64_t seenOfftargetAlready = (*ptrOfftargetFlag >> (signatureId % 64)) & 1ULL;


            if (!seenOfftargetAlready) {
                // Begin calculating MIT score
                if (calcMethod.mit) {
                    if (dist > 0) {
                        // totScoreMit += precalculatedScores[mismatches] * (double)occurrences;

                        // Calculate index of precalculatedScores
                        uint64_t mask_40bit = mismatches;
                        uint32_t mask_20bit = 0;
                        uint32_t shift = 0;

                        for (uint32_t bit = 0; bit < 20; bit++) {
                            mask_20bit |= (mask_40bit & 1) << shift;
                            mask_40bit >>= 2;
                            shift += 1;
                        }

                        atomicAdd(&totScoreMit, precalculatedScores[mask_20bit] * (double)occurrences);
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
                        
                        for (uint32_t pos = 0; pos < 20; pos++) {
                            uint32_t mask = pos << 4;
                            
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
                    // This seems ever so slighly faster when testing
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
                        continueCUDA = false;
                        checkNextSlice = false;
                    }
                    break;
                case ScoreMethod::mitOrCfd:
                    if (totScoreMit > maximum_sum ||
                        totScoreCfd > maximum_sum) {
                        continueCUDA = false;
                        checkNextSlice = false;
                    }
                    break;
                case ScoreMethod::avgMitCfd:
                    if (((totScoreMit + totScoreCfd) / 2.0) > maximum_sum) {
                        continueCUDA = false;
                        checkNextSlice = false;
                    }
                    break;
                case ScoreMethod::mit:
                    if (totScoreMit > maximum_sum) {
                        continueCUDA = false;
                        checkNextSlice = false;
                    }
                    break;
                case ScoreMethod::cfd:
                    if (totScoreCfd > maximum_sum) {
                        continueCUDA = false;
                        checkNextSlice = false;
                    }
                    break;
                default:
                    break;
                }

                if (!continueCUDA) {
                    return;
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
    unsigned int maxDist = atoi(argv[3]);
    
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

    // Create a CUDA stream for async operations, hardware also has a seperate queue for device-to-host
    cudaStream_t stream, streamD2H;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&streamD2H);
    
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
    // phmap::flat_hash_map<uint64_t, double> precalculatedScores;
    const uint_fast32_t precalculatedScoresSize = (1 << 20) - 1; // 20 bit mask
    vector<double> precalculatedScores(precalculatedScoresSize, -1.0);

    double* d_precalculatedScores;
    // This should take 8MB of VRAM, I don't think we will need to check free VRAM for this
    gpuErrChk(cudaMallocAsync(&d_precalculatedScores, precalculatedScoresSize * sizeof(double), stream));

    for (int i = 0; i < scoresCount; i++) {
        uint64_t mask = 0;
        double score = 0.0;
        fread(&mask, sizeof(uint64_t), 1, fp);
        fread(&score, sizeof(double), 1, fp);

        // Compress the 40-bit mask into a 20-bit mask for space efficiency
        uint64_t mask_40bit = mask;
        uint_fast32_t mask_20bit = 0;
        size_t shift = 0;

        for (size_t bit = 0; bit < 20; bit++) {
            mask_20bit |= (mask_40bit & 1) << shift;
            mask_40bit >>= 2;
            shift += 1;
        }
        
        // precalculatedScores.insert(pair<uint64_t, double>(mask, score));
        precalculatedScores[mask_20bit] = score;
    }

    gpuErrChk(cudaMemcpyAsync(d_precalculatedScores, precalculatedScores.data(), precalculatedScoresSize * sizeof(double), cudaMemcpyHostToDevice, stream));

    /** Load in all of the off-target sites */
    uint64_t* offtargets;

    {
        size_t cuda_mem_free, cuda_mem_total;
        cudaMemGetInfo(&cuda_mem_free, &cuda_mem_total);

        if (offtargetsCount > (cuda_mem_free/2)) {  // (/2) Leave space for the signatures
            // Not enough VRAM
            fprintf(stderr, "Not enough VRAM, using CUDA managed memory for offtargets\n");
            fprintf(stderr, "%zuMB Required\n", offtargetsCount * sizeof(uint64_t) / (1024*1024));

            gpuErrChk(cudaMallocManaged(&offtargets, offtargetsCount * sizeof(uint64_t)));

            if (fread(offtargets, sizeof(uint64_t), offtargetsCount, fp) == 0) {
                fprintf(stderr, "Error reading index: loading off-target sequences failed\n");
                return 1;
            }
        } else {
            // Enough VRAM, use device memory
            gpuErrChk(cudaMalloc(&offtargets, offtargetsCount * sizeof(uint64_t)));

            vector<uint64_t> offtargetsTemp(offtargetsCount);
            if (fread(offtargetsTemp.data(), sizeof(uint64_t), offtargetsCount, fp) == 0) {
                fprintf(stderr, "Error reading index: loading off-target sequences failed\n");
                return 1;
            }

            // Can't use Async here, since the vector will be destroyed after this scope
            gpuErrChk(cudaMemcpy(offtargets, offtargetsTemp.data(), offtargetsCount * sizeof(uint64_t), cudaMemcpyHostToDevice));
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
    uint64_t *d_someSignatures;  // Used when not enough VRAM
    bool signaturesLowMem;
    {
        size_t cuda_mem_free, cuda_mem_total;
        cudaMemGetInfo(&cuda_mem_free, &cuda_mem_total);

        size_t allSignaturesSize = seqCount * sliceCount * sizeof(uint64_t);
        signaturesLowMem = allSignaturesSize > cuda_mem_free;
        if (signaturesLowMem) {
            // Not enough VRAM
            fprintf(stderr, "Not enough VRAM, using small chunks for signatures\n");
            fprintf(stderr, "%zuMB Required\n", allSignaturesSize / (1024*1024));

            // cudaMallocManaged(&allSignatures, allSignaturesSize);
            allSignatures = (uint64_t*)malloc(allSignaturesSize);
            if (fread(allSignatures, sizeof(uint64_t), allSignaturesSize, fp) == 0) {
                fprintf(stderr, "Error reading index: reading slice contents failed\n");
                return 1;
            }
        } else {
            // Enough VRAM, use device memory
            gpuErrChk(cudaMalloc(&allSignatures, allSignaturesSize));

            vector<uint64_t> allSignaturesTemp(seqCount * sliceCount);
            if (fread(allSignaturesTemp.data(), sizeof(uint64_t), allSignaturesSize, fp) == 0) {
                fprintf(stderr, "Error reading index: reading slice contents failed\n");
                return 1;
            }
            // Can't use Async here, since the vector will be destroyed after this scope
            gpuErrChk(cudaMemcpy(allSignatures, allSignaturesTemp.data(), allSignaturesSize, cudaMemcpyHostToDevice));

            // TODO: Read in chunks
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

    uint64_t maxSliceSize = 0;

    uint64_t *offset = allSignatures;
    for (size_t i = 0; i < sliceCount; i++) {
        for (size_t j = 0; j < sliceLimit; j++) {
            size_t idx = i * sliceLimit + j;
            sliceLists[i][j] = offset;
            offset += allSlicelistSizes[idx];
            maxSliceSize = max(maxSliceSize, allSlicelistSizes[idx]);
        }
    }

    if (signaturesLowMem) {
        gpuErrChk(cudaMalloc(&d_someSignatures, maxSliceSize * sizeof(uint64_t) * sliceCount));

        fprintf(stderr, "Allocated %zuMB chunks for signatures\n", maxSliceSize * sizeof(uint64_t) * sliceCount / (1024*1024));
        // TODO: Figure out what happens if this doesn't fit in VRAM
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

    // Increase L1 cache size at the expense of shared memory (which we don't use)
    gpuErrChk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    // Find the best block size
    int gridSize, blockSize;
    gpuErrChk(cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, sequenceToSignatureCUDA, 0, queryCount));

    // Allocate VRAM for the query data
    uint8_t *d_queryDataSet;
    uint64_t *d_querySignatures;
    gpuErrChk(cudaMalloc(&d_queryDataSet, queryCount * (seqLength + 1) * sizeof(uint8_t)));
    gpuErrChk(cudaMalloc(&d_querySignatures, queryCount * sizeof(uint64_t)));

    // Copy the query data to the VRAM
    gpuErrChk(cudaMemcpy(d_queryDataSet, queryDataSet.data(), queryCount * (seqLength + 1) * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Calculate the signatures
    sequenceToSignatureCUDA<<<blockSize, gridSize>>>(queryCount, seqLength, d_queryDataSet, d_querySignatures);
    gpuErrChk(cudaGetLastError());

    // Wait for CUDA to finish
    gpuErrChk(cudaDeviceSynchronize());

    // Free VRAM and copy the signatures back to system RAM
    gpuErrChk(cudaFree(d_queryDataSet));
    gpuErrChk(cudaMemcpyAsync(querySignatures.data(), d_querySignatures, queryCount * sizeof(uint64_t), cudaMemcpyDeviceToHost, streamD2H));
    gpuErrChk(cudaFreeAsync(d_querySignatures, streamD2H));

    // Allocate VRAM for various variables
    double *d_cfdPamPenalties;
    double *d_cfdPosPenalties;
    gpuErrChk(cudaMallocAsync(&d_cfdPamPenalties, sizeof(cfdPamPenalties), stream));
    gpuErrChk(cudaMallocAsync(&d_cfdPosPenalties, sizeof(cfdPosPenalties), stream));
    gpuErrChk(cudaMemcpyAsync(d_cfdPamPenalties, cfdPamPenalties, sizeof(cfdPamPenalties), cudaMemcpyHostToDevice, stream));
    gpuErrChk(cudaMemcpyAsync(d_cfdPosPenalties, cfdPosPenalties, sizeof(cfdPosPenalties), cudaMemcpyHostToDevice, stream));

    uint64_t *d_offtargetToggles;
    gpuErrChk(cudaMallocAsync(&d_offtargetToggles, numOfftargetToggles * sizeof(uint64_t), stream));
    gpuErrChk(cudaMemsetAsync(d_offtargetToggles, 0, numOfftargetToggles * sizeof(uint64_t), stream));
    uint64_t *d_offtargetTogglesTail = d_offtargetToggles + numOfftargetToggles - 1;

    double *d_totScoreMit;
    double *d_totScoreCfd;
    gpuErrChk(cudaMallocAsync(&d_totScoreMit, sizeof(double), stream));
    gpuErrChk(cudaMallocAsync(&d_totScoreCfd, sizeof(double), stream));

    uint32_t *d_numOffTargetSitesScored;
    gpuErrChk(cudaMallocAsync(&d_numOffTargetSitesScored, sizeof(int), stream));

    // Mapped memory for checkNextSlice
    bool *h_checkNextSlice, *d_checkNextSlice;
    gpuErrChk(cudaHostAlloc(&h_checkNextSlice, sizeof(bool), cudaHostAllocMapped));
    gpuErrChk(cudaHostGetDevicePointer(&d_checkNextSlice, h_checkNextSlice, 0));

    // 256 byte limit on the number of arguments to a kernel, so we use structs
    // This one stays in constant memory however
    const struct constScoringArgs constant_args = {
        .offtargets = offtargets,
        .offtargetTogglesTail = d_offtargetTogglesTail,
        .maxDist = maxDist,
        .calcMethod = calcMethod,
        .scoreMethod = scoreMethod,
        .precalculatedScores = d_precalculatedScores,
        .cfdPamPenalties = d_cfdPamPenalties,
        .cfdPosPenalties = d_cfdPosPenalties,
        .totScoreMit = d_totScoreMit,
        .totScoreCfd = d_totScoreCfd,
        .numOffTargetSitesScored = d_numOffTargetSitesScored,
        .checkNextSlice = d_checkNextSlice,
    };
    // Copy to constant memory
    gpuErrChk(cudaMemcpyToSymbolAsync(d_constant_args, &constant_args, sizeof(struct constScoringArgs), 0, cudaMemcpyHostToDevice, stream));

    unordered_map<uint64_t, unordered_set<uint64_t>> searchResults;

    // Recalculate the block size for the scoring kernel
    gpuErrChk(cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, scoringCUDA, 0, maxSliceSize));

    /** For each candidate guide */
    for (size_t searchIdx = 0; searchIdx < querySignatures.size(); searchIdx++) {

        auto searchSignature = querySignatures[searchIdx];

        /** Global scores */
        cudaMemsetAsync(d_totScoreMit, 0, sizeof(double), stream);  // totScoreMit = 0.0;
        cudaMemsetAsync(d_totScoreCfd, 0, sizeof(double), stream);  // totScoreCfd = 0.0;
        
        cudaMemsetAsync(d_numOffTargetSitesScored, 0, sizeof(int), stream);  // numOffTargetSitesScored = 0;
        double maximum_sum = (10000.0 - threshold*100) / threshold;

        *h_checkNextSlice = true;

        vector<uint64_t> searchSlices(sliceCount);

        /** Prepare each ISSL slice */
        for (size_t i = 0; i < sliceCount; i++) {
            uint64_t sliceMask = sliceLimit - 1;
            size_t sliceShift = sliceWidth * i;
            sliceMask = sliceMask << sliceShift;
            
            uint64_t searchSlice = (searchSignature & sliceMask) >> sliceShift;
            searchSlices[i] = searchSlice;

            if (signaturesLowMem) {
                cudaMemcpyAsync(d_someSignatures + i * maxSliceSize, sliceLists[i][searchSlice],
                allSlicelistSizes[i * sliceLimit + searchSlice] * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
            }
        }

        /** Calculate each ISSL slice */
        for (size_t i = 0; i < sliceCount; i++) {
            uint64_t &searchSlice = searchSlices[i];
            
            size_t idx = i * sliceLimit + searchSlice;
            
            size_t signaturesInSlice = allSlicelistSizes[idx];

            uint64_t *sliceOffset = signaturesLowMem ? (d_someSignatures + i * maxSliceSize) : sliceLists[i][searchSlice];

            // Check last run to see if we should continue, also syncs the stream
            cudaDeviceSynchronize();
            if (!*h_checkNextSlice) {
                break;
            }

            scoringCUDA<<<blockSize, gridSize>>>(sliceOffset, signaturesInSlice, searchSignature, maximum_sum);
            // Continue next iteration to prepare the next run
        }
        cudaDeviceSynchronize();

        // memset(offtargetToggles.data(), 0, sizeof(uint64_t)*offtargetToggles.size());
        cudaMemsetAsync(d_offtargetToggles, 0, numOfftargetToggles * sizeof(uint64_t), stream);

        double totScoreMit = 0.0;
        double totScoreCfd = 0.0;
        cudaMemcpyAsync(&totScoreMit, d_totScoreMit, sizeof(double), cudaMemcpyDeviceToHost, streamD2H);
        cudaMemcpyAsync(&totScoreCfd, d_totScoreCfd, sizeof(double), cudaMemcpyDeviceToHost, streamD2H);

        cudaStreamSynchronize(streamD2H);
        querySignatureMitScores[searchIdx] = 10000.0 / (100.0 + totScoreMit);
        querySignatureCfdScores[searchIdx] = 10000.0 / (100.0 + totScoreCfd);
    }

    // Free VRAM
    gpuErrChk(cudaDeviceReset());

    /** Print global scores to stdout */
    for (size_t searchIdx = 0; searchIdx < querySignatures.size(); searchIdx++) {
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