#pragma once

#include "config.hpp"
#include "const_expr.hpp"

// ============================================================================
// BOP Configuration Parameters - Ported from ChampSim BOP Prefetcher
// ============================================================================
// Best Offset Pattern (BOP): Tests various offset candidates, tracks which
// offsets lead to cache hits (stored in Recency Ring), and prefetches based
// on the best-performing offsets discovered during each learning phase.

// BOP functional knobs
#define BOP_ENABLE_PREF_BUFFER 0    // Disable prefetch buffer for simplicity
#define BOP_SINGLE_PREFETCH 1       // Single prefetch per cycle for HLS efficiency

// Recency Ring (RR) parameters
#define BOP_RR_SIZE 256             // Size of recency ring buffer
#define BOP_RR_ADDR_BIT 32          // Address width for RR entries

// Pattern learning parameters
#define BOP_NUM_CANDIDATES 46       // Number of offset candidates to test
#define BOP_MAX_ROUNDS 100          // Max accesses before phase ends
#define BOP_MAX_SCORE 31            // Max score before phase ends
#define BOP_TOP_N 1                 // Number of best offsets to keep (prefetch degree)
#define BOP_CANDIDATE_WIDTH 8       // Bit width for candidate offset values

// HLS-specific buffer sizes
#define BOP_PREF_BUFFER_SIZE 256    // Prefetch buffer depth
#define BOP_PREF_DEGREE 4           // Prefetches issued per cycle

// Score tracking parameters
#define BOP_SCORE_BIT 6             // Bit width for score counters
#define BOP_SCORE_MAX ((1 << BOP_SCORE_BIT) - 1)
#define BOP_COUNTER_BIT 16          // Bit width for round counter

// Index bitwidth parameters (derived from dimensions)
#define BOP_RR_INDEX_BIT 9          // Bit width for recency ring indices (256 entries)
#define BOP_CANDIDATE_INDEX_BIT 6   // Bit width for candidate indices (44 candidates fit in 6 bits)
#define BOP_TOP_N_INDEX_BIT 2       // Bit width for top-N index (BOP_TOP_N <= 4)

// Memory layout constants
#define BOP_LOG2_PAGE_SIZE 12       // 4KB page size
#define BOP_LOG2_BLOCK_SIZE 6       // 64-byte cache line
#define BOP_PAGE_SIZE (1 << BOP_LOG2_PAGE_SIZE)
#define BOP_BLOCK_SIZE (1 << BOP_LOG2_BLOCK_SIZE)
#define BOP_PAGE_OFFSET_BITS (BOP_LOG2_PAGE_SIZE - BOP_LOG2_BLOCK_SIZE)
#define BOP_PAGE_OFFSET_MASK ((1 << BOP_PAGE_OFFSET_BITS) - 1)
#define BOP_BLOCK_OFFSET_MASK ((1 << BOP_LOG2_BLOCK_SIZE) - 1)

// Candidate offset values (from ChampSim BOP)
// Array of offsets to test: {1, -1, 2, -2, 3, -3, ..., 40, -40}
#define BOP_CANDIDATES_INIT {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,9,-9,10,-10,11,-11,12,-12,13,-13,14,-14,15,-15,16,-16,18,-18,20,-20,24,-24,30,-30,32,-32,36,-36,40,-40}
