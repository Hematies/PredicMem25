#pragma once

#include "bop_config.hpp"
#include "bop_data_type.hpp"

// ============================================================================
// BOP Recency Ring
// ============================================================================
// Tracks recently accessed cache line addresses to detect patterns.
// When testing a candidate offset, we search the RR to see if
// (current_address - offset) was recently accessed. If found, the offset
// is likely a good prefetch pattern.
//
// Implementation: FIFO buffer with circular replacement
// Template Parameters:
//   rr_entry_t: Type for RR entries (default: bop_rr_entry_t)
//   rr_index_t: Type for RR indices (default: bop_rr_index_t)

template <typename rr_entry_t = bop_rr_entry_t,
          typename rr_index_t = bop_rr_index_t>
class BOPRecencyRing {
public:
    // Member arrays
    rr_entry_t entries[BOP_RR_SIZE];                             // RR buffer entries
    rr_index_t head;                                              // Current write position (FIFO)

    // Constructor: trivial initialization
    BOPRecencyRing() = default;

    // ========================================================================
    // search_entry: Check if an address exists in the recency ring
    // ========================================================================
    // Returns true if the address is found, false otherwise
    // Time Complexity: O(BOP_RR_SIZE)
    bop_valid_t search_entry(rr_entry_t addr) {
        // #pragma HLS PIPELINE II=1
        // #pragma HLS UNROLL FACTOR=16
		#pragma HLS INLINE

        #pragma HLS ARRAY_PARTITION variable=entries complete 

        bop_valid_t found = 0;
        for (bop_rr_index_loop_t i = 0; i < BOP_RR_SIZE; i++) {
            #pragma HLS UNROLL
            if (entries[i] == addr) {
                found = 1;
            }
        }
        return found;
    }

    // ========================================================================
    // insert_entry: Insert address into recency ring with FIFO replacement
    // ========================================================================
    // Inserts address at current head, wraps around after reaching end
    void insert_entry(rr_entry_t addr) {
        // #pragma HLS PIPELINE II=1
		#pragma HLS INLINE
        #pragma HLS ARRAY_PARTITION variable=entries complete 

        entries[head] = addr;
        head = (head + 1) % BOP_RR_SIZE;
    }

    // ========================================================================
    // clear: Reset recency ring
    // ========================================================================
    void clear() {
        // #pragma HLS PIPELINE II=1
        // #pragma HLS UNROLL FACTOR=32
		#pragma HLS INLINE
        #pragma HLS ARRAY_PARTITION variable=entries complete 

        for (bop_rr_index_loop_t i = 0; i < BOP_RR_SIZE; i++) {
            #pragma HLS UNROLL
            entries[i] = 0;
        }
        head = 0;
    }
};
