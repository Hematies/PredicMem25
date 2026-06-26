#pragma once

#include "config.hpp"
#include "const_expr.hpp"

// ============================================================================
// SPP Configuration Parameters - Ported from ChampSim SPP
// ============================================================================

// SPP functional knobs
#define SPP_LOOKAHEAD_ON 1
#define SPP_FILTER_ON 0
#define SPP_GHR_ON 1

// Signature table parameters
#define SPP_ST_SET 256
#define SPP_ST_WAY 4
#define SPP_ST_TAG_BIT 16
#define SPP_ST_TAG_MASK ((1 << SPP_ST_TAG_BIT) - 1)
#define SPP_SIG_SHIFT 3
#define SPP_SIG_BIT 12
#define SPP_SIG_MASK ((1 << SPP_SIG_BIT) - 1)
#define SPP_SIG_DELTA_BIT 7

// Pattern table parameters
#define SPP_PT_SET 512
#define SPP_PT_WAY 4
#define SPP_C_SIG_BIT 4
#define SPP_C_DELTA_BIT 4
#define SPP_C_SIG_MAX ((1 << SPP_C_SIG_BIT) - 1)
#define SPP_C_DELTA_MAX ((1 << SPP_C_DELTA_BIT) - 1)

// Prefetch filter parameters
#define SPP_QUOTIENT_BIT 10
#define SPP_REMAINDER_BIT 6
#define SPP_HASH_BIT (SPP_QUOTIENT_BIT + SPP_REMAINDER_BIT + 1)
#define SPP_FILTER_SET (1 << SPP_QUOTIENT_BIT)
#define SPP_FILL_THRESHOLD 90
#define SPP_PF_THRESHOLD 25

// Global register parameters
#define SPP_GLOBAL_COUNTER_BIT 10
#define SPP_GLOBAL_COUNTER_MAX ((1 << SPP_GLOBAL_COUNTER_BIT) - 1)
#define SPP_MAX_GHR_ENTRY 8

// HLS-specific buffer sizes
#define SPP_MAX_PREFETCH_QUEUE 64

// Memory layout constants
#define SPP_LOG2_PAGE_SIZE 12
#define SPP_PAGE_SIZE (1 << SPP_LOG2_PAGE_SIZE)
#define SPP_LOG2_BLOCK_SIZE BLOCK_SIZE_LOG2
#define SPP_BLOCK_SIZE BLOCK_SIZE
