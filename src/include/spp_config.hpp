#pragma once

#include "config.hpp"
#include "const_expr.hpp"

// ============================================================================
// SPP Configuration Parameters - Ported from ChampSim SPP
// ============================================================================

// SPP functional knobs
constexpr bool SPP_LOOKAHEAD_ON = true;
constexpr bool SPP_FILTER_ON = true;
constexpr bool SPP_GHR_ON = true;

// Signature table parameters
constexpr uint32_t SPP_ST_SET = 1;
constexpr uint32_t SPP_ST_WAY = 256;
constexpr uint32_t SPP_ST_TAG_BIT = 16;
constexpr uint32_t SPP_ST_TAG_MASK = ((1 << SPP_ST_TAG_BIT) - 1);
constexpr uint32_t SPP_SIG_SHIFT = 3;
constexpr uint32_t SPP_SIG_BIT = 12;
constexpr uint32_t SPP_SIG_MASK = ((1 << SPP_SIG_BIT) - 1);
constexpr uint32_t SPP_SIG_DELTA_BIT = 7;

// Pattern table parameters
constexpr uint32_t SPP_PT_SET = 512;
constexpr uint32_t SPP_PT_WAY = 4;
constexpr uint32_t SPP_C_SIG_BIT = 4;
constexpr uint32_t SPP_C_DELTA_BIT = 4;
constexpr uint32_t SPP_C_SIG_MAX = ((1 << SPP_C_SIG_BIT) - 1);
constexpr uint32_t SPP_C_DELTA_MAX = ((1 << SPP_C_DELTA_BIT) - 1);

// Prefetch filter parameters
constexpr uint32_t SPP_QUOTIENT_BIT = 10;
constexpr uint32_t SPP_REMAINDER_BIT = 6;
constexpr uint32_t SPP_HASH_BIT = (SPP_QUOTIENT_BIT + SPP_REMAINDER_BIT + 1);
constexpr uint32_t SPP_FILTER_SET = (1 << SPP_QUOTIENT_BIT);
constexpr uint32_t SPP_FILL_THRESHOLD = 90;
constexpr uint32_t SPP_PF_THRESHOLD = 25;

// Global register parameters
constexpr uint32_t SPP_GLOBAL_COUNTER_BIT = 10;
constexpr uint32_t SPP_GLOBAL_COUNTER_MAX = ((1 << SPP_GLOBAL_COUNTER_BIT) - 1);
constexpr uint32_t SPP_MAX_GHR_ENTRY = 8;

// HLS-specific buffer sizes
constexpr uint32_t SPP_MAX_PREFETCH_QUEUE = 64;

// Memory layout constants
constexpr uint32_t SPP_LOG2_PAGE_SIZE = 12;
constexpr uint32_t SPP_PAGE_SIZE = (1 << SPP_LOG2_PAGE_SIZE);
constexpr uint32_t SPP_LOG2_BLOCK_SIZE = BLOCK_SIZE_LOG2;
constexpr uint32_t SPP_BLOCK_SIZE = BLOCK_SIZE;
