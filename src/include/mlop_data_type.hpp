#pragma once

#include "config.hpp"
#include "const_expr.hpp"
#include "data_type.hpp"
#include "mlop_config.hpp"

// ============================================================================
// MLOP Data Types
// ============================================================================
// Centralized type definitions for MLOP prefetcher components.
// All bitwidths are derived from configuration constants in mlop_config.hpp
// for consistency and easy parametrization across the entire design.
//
// Supports dual compilation modes:
// - CSIM_DEBUG: Uses standard C++ types for simulation
// - HLS: Uses ap_uint/ap_int fixed-width types for synthesis
//
// Key Parameters Used (from mlop_config.hpp):
// - MLOP_BLOCKS_IN_ZONE (64): Blocks per zone
// - MLOP_NUM_OFFSETS (127): Number of possible offsets
// - MLOP_PF_DEGREE (16): Number of prefetch degrees
// - MLOP_ZONE_ADDR_BIT (20): Zone address width
// - MLOP_ZONE_OFFSET_BIT (6): Zone offset width
// - MLOP_SCORE_BIT (10): Score counter width
// - MLOP_OFFSET_BIT (7): Offset value width
// - MLOP_DEGREE_BIT (4): Degree index width
// - MLOP_STATE_BIT (2): State value width

#ifdef CSIM_DEBUG
    // C Simulation mode (for debugging and testing)
    using mlop_address_t = uint64_t;                            // Full memory address
    using mlop_block_address_t = uint32_t;                      // Cache line address
    using mlop_zone_address_t = uint32_t;                       // Zone address
    using mlop_zone_offset_t = uint8_t;                         // Offset within zone (0-63)
    using mlop_offset_t = int8_t;                               // Signed offset (-63 to +63)
    using mlop_score_t = uint16_t;                              // Score counter
    using mlop_degree_t = uint8_t;                              // Degree index
    using mlop_state_t = uint8_t;                               // Block state (0-2)
    using mlop_counter_t = uint32_t;                            // General counter
    using mlop_valid_t = bool;                                  // Valid bit
    
#else
    // HLS Synthesis mode (uses fixed-width ap_int types)
    // Bitwidths derived from mlop_config.hpp constants
    using mlop_address_t = ap_uint<32>;                         // Address (32-bit for typical caches)
    using mlop_block_address_t = ap_uint<32 - MLOP_LOG2_BLOCK_SIZE>;  // Block address
    using mlop_zone_address_t = ap_uint<MLOP_ZONE_ADDR_BIT>;    // Zone number
    using mlop_zone_offset_t = ap_uint<MLOP_ZONE_OFFSET_BIT>;   // Zone offset (6 bits for 64 blocks)
    using mlop_offset_t = ap_int<MLOP_OFFSET_BIT>;              // Signed offset (-63 to +63)
    using mlop_score_t = ap_uint<MLOP_SCORE_BIT>;               // Score counter (10 bits for 500+)
    using mlop_degree_t = ap_uint<MLOP_DEGREE_BIT>;             // Degree index (4 bits for 16 degrees)
    using mlop_state_t = ap_uint<MLOP_STATE_BIT>;               // State (2 bits: 0-2)
    using mlop_counter_t = ap_uint<MLOP_COUNTER_BIT>;           // General counter
    using mlop_valid_t = ap_uint<1>;                            // 1-bit valid flag
#endif

// ============================================================================
// Type Index Definitions - For loop counters and array indices
// ============================================================================

#ifdef CSIM_DEBUG
    using mlop_degree_loop_t = uint8_t;                         // Loop index for degrees
    using mlop_offset_loop_t = uint16_t;                        // Loop index for offsets
    using mlop_zone_offset_loop_t = uint8_t;                    // Loop index for zone blocks
#else
    using mlop_degree_loop_t = ap_uint<MLOP_DEGREE_BIT>;        // Loop index for degrees
    using mlop_offset_loop_t = ap_uint<11>;                     // Loop index for offsets (max 127)
    using mlop_zone_offset_loop_t = ap_uint<MLOP_ZONE_OFFSET_BIT>;  // Loop index for zone blocks
#endif

// ============================================================================
// MLOP Threshold Types
// ============================================================================

#ifdef CSIM_DEBUG
    using mlop_threshold_t = uint16_t;                          // Threshold value (ratio * NUM_UPDATES)
#else
    using mlop_threshold_t = ap_uint<MLOP_SCORE_BIT>;           // Threshold (same width as score)
#endif



