#pragma once

#include "bop_config.hpp"
#include "bop_data_type.hpp"

// ============================================================================
// BOP Prefetch Buffer
// ============================================================================
// Optional buffer for staging prefetch requests before issuing them to L2.
// Allows throttling prefetch rate and managing MSHR congestion.
//
// In the HLS implementation, this is typically disabled (BOP_ENABLE_PREF_BUFFER=0)
// to simplify the pipeline. When enabled, it provides decoupling between
// prefetch generation and issuance.
//
// Template Parameters:
//   address_t: Type for addresses (default: bop_address_t)
//   index_t: Type for buffer indices (default: bop_rr_index_t)

template <typename address_t = bop_address_t,
          typename index_t = bop_rr_index_t>
class BOPPrefetchBuffer {
public:
    // Member arrays
    address_t buffer[BOP_PREF_BUFFER_SIZE];                       // Prefetch buffer
    index_t head;                                                  // Write position
    index_t tail;                                                  // Read position
    index_t count;                                                 // Number of valid entries

    // Constructor: trivial initialization
    BOPPrefetchBuffer() = default;

    // ========================================================================
    // is_full: Check if buffer is at capacity
    // ========================================================================
    bop_valid_t is_full() {
        #pragma HLS PIPELINE II=1

        return (count >= BOP_PREF_BUFFER_SIZE) ? 1 : 0;
    }

    // ========================================================================
    // is_empty: Check if buffer is empty
    // ========================================================================
    bop_valid_t is_empty() {
        #pragma HLS PIPELINE II=1

        return (count == 0) ? 1 : 0;
    }

    // ========================================================================
    // buffer_prefetch: Add prefetch request to buffer
    // ========================================================================
    // Returns 1 if successfully buffered, 0 if buffer is full
    bop_valid_t buffer_prefetch(address_t addr) {
        #pragma HLS PIPELINE II=1

        if (count < BOP_PREF_BUFFER_SIZE) {
            buffer[head] = addr;
            head = (head + 1) % BOP_PREF_BUFFER_SIZE;
            count = count + 1;
            return 1;
        }
        return 0;
    }

    // ========================================================================
    // get_prefetch: Get next prefetch from buffer without removing
    // ========================================================================
    address_t get_prefetch() {
        #pragma HLS PIPELINE II=1

        return buffer[tail];
    }

    // ========================================================================
    // pop_prefetch: Remove and return next prefetch from buffer
    // ========================================================================
    // Only call when buffer is not empty
    address_t pop_prefetch() {
        #pragma HLS PIPELINE II=1

        address_t addr = buffer[tail];
        tail = (tail + 1) % BOP_PREF_BUFFER_SIZE;
        if (count > 0) {
            count = count - 1;
        }
        return addr;
    }

    // ========================================================================
    // clear: Reset buffer
    // ========================================================================
    void clear() {
        #pragma HLS PIPELINE II=1

        head = 0;
        tail = 0;
        count = 0;
    }
};
