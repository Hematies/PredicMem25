#pragma once

#include "config.hpp"
#include "const_expr.hpp"

// ============================================================================
// MLOP Configuration Parameters - Ported from ChampSim MLOP Prefetcher
// ============================================================================
// Multi-Lookahead Offset Prefetcher (MLOP): Tracks access patterns within
// cache zones (pages) and learns which offsets lead to productive prefetches.
// Uses an access map table to maintain state of cache blocks and a scoring
// mechanism to identify best offsets for prefetching across multiple degrees.
//
// Key Algorithm Concepts:
// - Zone: A cache zone (page) containing multiple blocks
// - Access Map: Tracks state (INIT, ACCESS, PREFETCH) of each block in zone
// - Offset: Distance between current block and prefetched block (range: -63 to +63)
// - Degree: Prefetch degree/lookahead level
// - Learning Round: After NUM_UPDATES accesses, selects best offsets

// ============================================================================
// MLOP Functional Parameters
// ============================================================================

// Prefetching configuration
#define MLOP_SINGLE_PREFETCH 1         // Single prefetch per cycle for HLS efficiency (1=yes, 0=multi-prefetch)
#define MLOP_PF_DEGREE 1              // Number of prefetch degrees (lookahead levels)
#define MLOP_NUM_UPDATES 500           // Accesses per learning round
#define MLOP_L1D_THRESH 2.0            // L1D score threshold (ratio to NUM_UPDATES)
#define MLOP_L2C_THRESH 0.75           // L2C score threshold
#define MLOP_LLC_THRESH 0.3            // LLC score threshold
#define MLOP_DEBUG_LEVEL 0             // Debug verbosity (0=off)

// ============================================================================
// Zone and Block Configuration
// ============================================================================

// Zone (page) structure: 4KB pages with 64-byte cache lines = 64 blocks/zone
#define MLOP_LOG2_PAGE_SIZE 12         // 4KB page size
#define MLOP_LOG2_BLOCK_SIZE 6         // 64-byte cache lines
#define MLOP_PAGE_SIZE (1 << MLOP_LOG2_PAGE_SIZE)
#define MLOP_BLOCK_SIZE (1 << MLOP_LOG2_BLOCK_SIZE)
#define MLOP_BLOCKS_IN_ZONE (MLOP_PAGE_SIZE / MLOP_BLOCK_SIZE)  // 64 blocks

// Offset range: from -(BLOCKS_IN_ZONE-1) to +(BLOCKS_IN_ZONE-1)
#define MLOP_MAX_OFFSET (MLOP_BLOCKS_IN_ZONE - 1)  // +63
#define MLOP_MIN_OFFSET (-(MLOP_BLOCKS_IN_ZONE - 1))  // -63
#define MLOP_NUM_OFFSETS (2 * MLOP_BLOCKS_IN_ZONE - 1)  // 127 possible offsets
#define MLOP_ORIGIN (MLOP_BLOCKS_IN_ZONE - 1)  // Index for offset 0 (center)

// Access map table sizing
// A reasonable size for typical cache hierarchies (32 blocks/zone typical)
#define MLOP_AMT_SIZE 32               // Number of zone entries in access map table

// History queue configuration
#define MLOP_QUEUE_SIZE (MLOP_PF_DEGREE - 1)  // Queue depth for tracking recent accesses

// ============================================================================
// Threshold Calculation (from knobs)
// ============================================================================
// These are computed from the ratio knobs multiplied by NUM_UPDATES
#define MLOP_L1D_THRESH_SCORE ((uint32_t)(MLOP_L1D_THRESH * MLOP_NUM_UPDATES))
#define MLOP_L2C_THRESH_SCORE ((uint32_t)(MLOP_L2C_THRESH * MLOP_NUM_UPDATES))
#define MLOP_LLC_THRESH_SCORE ((uint32_t)(MLOP_LLC_THRESH * MLOP_NUM_UPDATES))

// ============================================================================
// HLS-Specific Parameters
// ============================================================================

// Buffer and array sizes for HLS implementation
#define MLOP_MAX_PREFETCH_QUEUE 64     // Maximum prefetches queued per cycle
#define MLOP_OFFSET_ARRAY_SIZE MLOP_NUM_OFFSETS  // Score array dimension

// ============================================================================
// State Encoding
// ============================================================================
// Access map block states (2 bits each):
// INIT     = 0: Block not yet accessed/prefetched
// ACCESS   = 1: Block accessed (cache hit/miss trigger)
// PREFETCH = 2: Block was prefetched

#define MLOP_STATE_INIT 0
#define MLOP_STATE_ACCESS 1
#define MLOP_STATE_PREFETCH 2

// ============================================================================
// HLS Pipeline and Synthesis Directives
// ============================================================================

// Target II (initiation interval) for main prefetcher loop
#define MLOP_TARGET_II 1               // Aim for single-cycle throughput

// ============================================================================
// Derived Constants
// ============================================================================

// Bitwidths derived from configuration (for HLS type sizing)
#define MLOP_ZONE_ADDR_BIT 20          // Zone number bitwidth (derived from address space)
#define MLOP_ZONE_OFFSET_BIT 6         // Zone offset bitwidth (log2(64))
#define MLOP_SCORE_BIT 10              // Score counter bitwidth (can count up to NUM_UPDATES)
#define MLOP_OFFSET_BIT 7              // Offset value bitwidth (signed: -63 to +63)
#define MLOP_DEGREE_BIT 4              // Degree bitwidth (log2(16))
#define MLOP_COUNTER_BIT 16            // General counter bitwidth
#define MLOP_STATE_BIT 2               // State bitwidth (3 states = 2 bits)

// ============================================================================
// Threshold Pre-computation (Constexpr Static)
// ============================================================================

// Pre-computed thresholds (computed at compile-time)
#define MLOP_L1D_THRESHOLD (mlop_threshold_t)(MLOP_L1D_THRESH * MLOP_NUM_UPDATES)
#define MLOP_L2C_THRESHOLD (mlop_threshold_t)(MLOP_L2C_THRESH * MLOP_NUM_UPDATES)
#define MLOP_LLC_THRESHOLD (mlop_threshold_t)(MLOP_LLC_THRESH * MLOP_NUM_UPDATES)

