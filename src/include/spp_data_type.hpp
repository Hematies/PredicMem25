#pragma once

#include "config.hpp"
#include "const_expr.hpp"
#include "data_type.hpp"

// ============================================================================
// SPP Data Types
// ============================================================================
// Comprehensive type definitions for the SPP prefetcher implementation
// Following the pattern from data_type.hpp for consistency with PredicMem25
//
// These types serve as default template arguments for SPP components
// and can be used directly or specialized for custom implementations

#ifdef CSIM_DEBUG
// Simulation types (for C-sim debugging)
typedef uint64_t spp_address_t;
typedef uint64_t spp_block_address_t;
typedef uint64_t spp_st_tag_t;
typedef uint64_t spp_st_sig_t;
typedef uint64_t spp_st_confidence_t;
typedef uint64_t spp_st_lru_t;
typedef int64_t spp_pt_delta_t;
typedef uint64_t spp_pt_confidence_t;
typedef uint64_t spp_filter_tag_t;
typedef uint64_t spp_filter_quotient_t;
typedef uint64_t spp_filter_remainder_t;
typedef uint64_t spp_ghr_counter_t;
typedef uint64_t spp_ghr_offset_t;
typedef uint64_t spp_ghr_valid_t;
typedef uint64_t spp_page_offset_t;
typedef uint64_t spp_page_t;
typedef uint64_t spp_sig_delta_t;
typedef uint64_t spp_prefetch_confidence_t;
typedef uint64_t spp_accuracy_t;
typedef uint64_t spp_st_set_index_t;
typedef uint64_t spp_st_way_index_t;
typedef uint64_t spp_pt_set_index_t;
typedef uint64_t spp_pt_way_index_t;
typedef uint64_t spp_filter_index_t;
typedef uint64_t spp_ghr_way_index_t;

#else
// HLS types (for synthesis)

// ============================================================================
// Core Address Types
// ============================================================================
typedef ap_uint<NUM_ADDRESS_BITS> spp_address_t;
typedef ap_uint<NUM_BLOCK_ADDRESS_BITS> spp_block_address_t;

// ============================================================================
// Signature Table (ST) Types
// ============================================================================
typedef ap_uint<SPP_ST_TAG_BIT> spp_st_tag_t;           // Page tag for matching
typedef ap_uint<SPP_SIG_BIT> spp_st_sig_t;              // Signature value
typedef ap_uint<SPP_C_SIG_BIT> spp_st_confidence_t;     // Signature confidence counter
typedef ap_uint<8> spp_st_lru_t;                        // LRU counter (max 256 ways)

// ============================================================================
// Pattern Table (PT) Types
// ============================================================================
typedef ap_int<SPP_SIG_DELTA_BIT> spp_pt_delta_t;       // Signed delta (7-bit sign-magnitude)
typedef ap_uint<32> spp_pt_confidence_t;   // Delta confidence counter

// ============================================================================
// Prefetch Filter Types
// ============================================================================
typedef ap_uint<SPP_REMAINDER_BIT> spp_filter_tag_t;    // Remainder tag for filtering

// ============================================================================
// Global History Register (GHR) Types
// ============================================================================
typedef ap_uint<SPP_GLOBAL_COUNTER_BIT> spp_ghr_counter_t;  // Global counters
typedef ap_uint<6> spp_ghr_offset_t;                         // Page offset for matching

// ============================================================================
// Helper / Utility Types
// ============================================================================
typedef ap_uint<1> spp_ghr_valid_t;                     // Entry valid bit
typedef ap_uint<6> spp_page_offset_t;                   // Cache line offset within page
typedef ap_int<7> spp_sig_delta_t;                      // 7-bit sign-magnitude representation

// ============================================================================
// Index and Size Types
// ============================================================================
typedef ap_uint<bitsNeeded(SPP_ST_SET)> spp_st_set_index_t;     // ST set index
typedef ap_uint<bitsNeeded(SPP_ST_WAY)> spp_st_way_index_t;     // ST way index
typedef ap_uint<bitsNeeded(SPP_PT_SET)> spp_pt_set_index_t;     // PT set index
typedef ap_uint<bitsNeeded(SPP_PT_WAY)> spp_pt_way_index_t;     // PT way index
typedef ap_uint<bitsNeeded(SPP_FILTER_SET)> spp_filter_index_t; // Filter index
typedef ap_uint<bitsNeeded(SPP_MAX_GHR_ENTRY)> spp_ghr_way_index_t; // GHR way index

// ============================================================================
// Output and Statistics Types
// ============================================================================
typedef ap_uint<SPP_GLOBAL_COUNTER_BIT> spp_accuracy_t; // Global accuracy percentage

#endif
