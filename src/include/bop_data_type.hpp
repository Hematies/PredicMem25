#pragma once

#include "config.hpp"
#include "const_expr.hpp"
#include "data_type.hpp"
#include "bop_config.hpp"

// ============================================================================
// BOP Data Types
// ============================================================================
// Centralized type definitions for BOP prefetcher components.
// Supports dual compilation modes:
// - CSIM_DEBUG: Uses standard C++ types for simulation
// - HLS: Uses ap_uint/ap_int fixed-width types for synthesis

#ifdef CSIM_DEBUG
    // C Simulation mode (for debugging and testing)
    using bop_address_t = uint64_t;                              // Full memory address
    using bop_block_address_t = uint32_t;                        // Cache line address (address >> LOG2_BLOCK_SIZE)
    using bop_rr_entry_t = uint32_t;                             // Recency ring entry (block address)
    using bop_candidate_t = int8_t;                              // Candidate offset (can be negative)
    using bop_candidate_index_t = uint8_t;                       // Index into candidates array
    using bop_score_t = uint8_t;                                 // Score counter for offset
    using bop_offset_t = int8_t;                                 // Cache line offset within page
    using bop_page_t = uint32_t;                                 // Page address
    using bop_valid_t = bool;                                    // Valid bit
    using bop_rr_index_t = uint16_t;                             // Index into recency ring
    using bop_counter_t = uint32_t;                              // Counter for round tracking
    
#else
    // HLS Synthesis mode (uses fixed-width ap_int types)
    using bop_address_t = ap_uint<32>;                           // 32-bit address
    using bop_block_address_t = ap_uint<26>;                     // Block address (32 - 6 bits)
    using bop_rr_entry_t = ap_uint<26>;                          // RR entry (block address)
    using bop_candidate_t = ap_int<8>;                           // Signed candidate offset
    using bop_candidate_index_t = ap_uint<6>;                    // 6-bit index (max 44 candidates)
    using bop_score_t = ap_uint<BOP_SCORE_BIT>;                  // Score counter
    using bop_offset_t = ap_int<7>;                              // 7-bit signed offset (-64 to +63)
    using bop_page_t = ap_uint<20>;                              // Page address (32 - 12 bits)
    using bop_valid_t = ap_uint<1>;                              // 1-bit valid
    using bop_rr_index_t = ap_uint<9>;                           // 9-bit index (max 512 entries, uses 256)
    using bop_counter_t = ap_uint<BOP_COUNTER_BIT>;              // Counter for round tracking
#endif

// ============================================================================
// Type Index Definitions - For loop counters and array indices
// ============================================================================

#ifdef CSIM_DEBUG
    using bop_rr_index_loop_t = uint16_t;                        // Loop index for RR iteration
    using bop_candidate_loop_t = uint8_t;                        // Loop index for candidate iteration
    using bop_top_n_index_t = uint8_t;                           // Index for selecting top-N offsets
#else
    using bop_rr_index_loop_t = ap_uint<9>;                      // Loop index for RR (max 256 entries)
    using bop_candidate_loop_t = ap_uint<6>;                     // Loop index for candidates (max 44)
    using bop_top_n_index_t = ap_uint<2>;                        // Loop index for top-N (max 4)
#endif

