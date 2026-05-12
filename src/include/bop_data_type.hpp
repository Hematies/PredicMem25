#pragma once

#include "config.hpp"
#include "const_expr.hpp"
#include "data_type.hpp"
#include "bop_config.hpp"

// ============================================================================
// BOP Data Types
// ============================================================================
// Centralized type definitions for BOP prefetcher components.
// All bitwidths are derived from configuration constants in bop_config.hpp
// for consistency and easy parametrization across the entire design.
//
// Supports dual compilation modes:
// - CSIM_DEBUG: Uses standard C++ types for simulation
// - HLS: Uses ap_uint/ap_int fixed-width types for synthesis
//
// Key Parameters Used (from bop_config.hpp):
// - BOP_RR_ADDR_BIT (32): Address bitwidth
// - BOP_LOG2_BLOCK_SIZE (6): Cache line size exponent
// - BOP_LOG2_PAGE_SIZE (12): Page size exponent
// - BOP_CANDIDATE_WIDTH (8): Candidate offset bitwidth
// - BOP_SCORE_BIT (6): Score counter bitwidth
// - BOP_COUNTER_BIT (16): Round counter bitwidth
// - BOP_RR_INDEX_BIT (9): RR index bitwidth
// - BOP_CANDIDATE_INDEX_BIT (6): Candidate index bitwidth
// - BOP_TOP_N_INDEX_BIT (2): Top-N index bitwidth

#ifdef CSIM_DEBUG
    // C Simulation mode (for debugging and testing)
    using bop_address_t = uint64_t;                              // Full memory address
    using bop_block_address_t = uint32_t;                        // Cache line address (address >> BOP_LOG2_BLOCK_SIZE)
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
    // Bitwidths derived from bop_config.hpp constants
    using bop_address_t = ap_uint<BOP_RR_ADDR_BIT>;              // Address (from BOP_RR_ADDR_BIT)
    using bop_block_address_t = ap_uint<BOP_RR_ADDR_BIT - BOP_LOG2_BLOCK_SIZE>;  // Block address
    using bop_rr_entry_t = ap_uint<BOP_RR_ADDR_BIT - BOP_LOG2_BLOCK_SIZE>;       // RR entry (block address)
    using bop_candidate_t = ap_int<BOP_CANDIDATE_WIDTH>;         // Signed candidate offset
    using bop_candidate_index_t = ap_uint<BOP_CANDIDATE_INDEX_BIT>;  // Index into candidates (from config)
    using bop_score_t = ap_uint<BOP_SCORE_BIT>;                  // Score counter (from BOP_SCORE_BIT)
    using bop_offset_t = ap_int<BOP_CANDIDATE_WIDTH + 1>;        // Signed offset (candidate width + 1)
    using bop_page_t = ap_uint<BOP_LOG2_PAGE_SIZE>;              // Page address (from BOP_LOG2_PAGE_SIZE)
    using bop_valid_t = ap_uint<1>;                              // 1-bit valid flag
    using bop_rr_index_t = ap_uint<BOP_RR_INDEX_BIT>;            // Index for RR (from BOP_RR_INDEX_BIT)
    using bop_counter_t = ap_uint<BOP_COUNTER_BIT>;              // Round counter (from BOP_COUNTER_BIT)
#endif

// ============================================================================
// Type Index Definitions - For loop counters and array indices
// ============================================================================

#ifdef CSIM_DEBUG
    using bop_rr_index_loop_t = uint16_t;                        // Loop index for RR iteration
    using bop_candidate_loop_t = uint8_t;                        // Loop index for candidate iteration
    using bop_top_n_index_t = uint8_t;                           // Index for selecting top-N offsets
#else
    using bop_rr_index_loop_t = ap_uint<BOP_RR_INDEX_BIT>;       // Loop index for RR (from BOP_RR_INDEX_BIT)
    using bop_candidate_loop_t = ap_uint<BOP_CANDIDATE_INDEX_BIT>;  // Loop index for candidates (from config)
    using bop_top_n_index_t = ap_uint<BOP_TOP_N_INDEX_BIT>;      // Loop index for top-N (from config)
#endif

