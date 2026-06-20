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
    typedef uint64_t bop_address_t;                      // Full memory address
    typedef uint32_t bop_block_address_t;                // Cache line address (address >> BOP_LOG2_BLOCK_SIZE)
    typedef uint32_t bop_rr_entry_t;                     // Recency ring entry (block address)
    typedef int8_t bop_candidate_t;                      // Candidate offset (can be negative)
    typedef uint8_t bop_candidate_index_t;               // Index into candidates array
    typedef uint8_t bop_score_t;                         // Score counter for offset
    typedef int8_t bop_offset_t;                         // Cache line offset within page
    typedef uint32_t bop_page_t;                         // Page address
    typedef bool bop_valid_t;                            // Valid bit
    typedef uint16_t bop_rr_index_t;                     // Index into recency ring
    typedef uint32_t bop_counter_t;                      // Counter for round tracking
    
#else
    // HLS Synthesis mode (uses fixed-width ap_int types)
    // Bitwidths derived from bop_config.hpp constants
    typedef ap_uint<BOP_RR_ADDR_BIT> bop_address_t;              // Address (from BOP_RR_ADDR_BIT)
    typedef ap_uint<BOP_RR_ADDR_BIT - BOP_LOG2_BLOCK_SIZE> bop_block_address_t;  // Block address
    typedef ap_uint<BOP_RR_ADDR_BIT - BOP_LOG2_BLOCK_SIZE> bop_rr_entry_t;       // RR entry (block address)
    typedef ap_int<BOP_CANDIDATE_WIDTH> bop_candidate_t;         // Signed candidate offset
    typedef ap_uint<BOP_CANDIDATE_INDEX_BIT> bop_candidate_index_t;  // Index into candidates (from config)
    typedef ap_uint<BOP_SCORE_BIT> bop_score_t;                  // Score counter (from BOP_SCORE_BIT)
    typedef ap_int<BOP_CANDIDATE_WIDTH + 1> bop_offset_t;        // Signed offset (candidate width + 1)
    typedef ap_uint<BOP_LOG2_PAGE_SIZE> bop_page_t;              // Page address (from BOP_LOG2_PAGE_SIZE)
    typedef ap_uint<1> bop_valid_t;                              // 1-bit valid flag
    typedef ap_uint<BOP_RR_INDEX_BIT> bop_rr_index_t;            // Index for RR (from BOP_RR_INDEX_BIT)
    typedef ap_uint<BOP_COUNTER_BIT> bop_counter_t;              // Round counter (from BOP_COUNTER_BIT)
#endif

// ============================================================================
// Type Index Definitions - For loop counters and array indices
// ============================================================================

#ifdef CSIM_DEBUG
    typedef uint16_t bop_rr_index_loop_t;                        // Loop index for RR iteration
    typedef uint8_t bop_candidate_loop_t;                        // Loop index for candidate iteration
    typedef uint8_t bop_top_n_index_t;                           // Index for selecting top-N offsets
#else
    typedef ap_uint<BOP_RR_INDEX_BIT> bop_rr_index_loop_t;       // Loop index for RR (from BOP_RR_INDEX_BIT)
    typedef ap_uint<BOP_CANDIDATE_INDEX_BIT> bop_candidate_loop_t;  // Loop index for candidates (from config)
    typedef ap_uint<BOP_TOP_N_INDEX_BIT> bop_top_n_index_t;      // Loop index for top-N (from config)
#endif
