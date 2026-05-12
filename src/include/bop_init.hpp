#pragma once

#include "bop_config.hpp"
#include "bop_data_type.hpp"
#include "bop_recency_ring.hpp"
#include "bop_pattern_learner.hpp"

// ============================================================================
// BOP Initialization Data Structures and Functions
// ============================================================================
// Constexpr-compatible initialization for BOP components.
// These structures and functions enable static compile-time initialization
// of arrays that can be reshaped/partitioned by HLS pragmas.

// ============================================================================
// Recency Ring Matrix Structure
// ============================================================================
struct BOPRecencyRingMatrix {
    bop_rr_entry_t entries[BOP_RR_SIZE];
    bop_rr_index_t head;

    // Constructor for constexpr initialization
    constexpr BOPRecencyRingMatrix() : head(0) {
        for (int i = 0; i < BOP_RR_SIZE; i++) {
            entries[i] = 0;
        }
    }
};

// ============================================================================
// Pattern Learner Matrix Structure
// ============================================================================
struct BOPPatternLearnerMatrix {
    bop_score_t scores[BOP_NUM_CANDIDATES];
    bop_candidate_loop_t candidate_ptr;
    bop_counter_t round_counter;
    bop_candidate_t best_offsets[BOP_TOP_N];
    uint32_t num_best_offsets;

    // Constructor for constexpr initialization
    constexpr BOPPatternLearnerMatrix() 
        : candidate_ptr(0), round_counter(0), num_best_offsets(0) {
        for (int i = 0; i < BOP_NUM_CANDIDATES; i++) {
            scores[i] = 0;
        }
        for (int i = 0; i < BOP_TOP_N; i++) {
            best_offsets[i] = 0;
        }
    }
};

// ============================================================================
// Prefetch Buffer Matrix Structure
// ============================================================================
struct BOPPrefetchBufferMatrix {
    bop_address_t buffer[BOP_PREF_BUFFER_SIZE];
    bop_rr_index_t head;
    bop_rr_index_t tail;
    bop_rr_index_t count;

    // Constructor for constexpr initialization
    constexpr BOPPrefetchBufferMatrix() : head(0), tail(0), count(0) {
        for (int i = 0; i < BOP_PREF_BUFFER_SIZE; i++) {
            buffer[i] = 0;
        }
    }
};

// ============================================================================
// Candidate Offsets Storage
// ============================================================================
struct BOPCandidateOffsets {
    bop_candidate_t values[BOP_NUM_CANDIDATES];

    // Constructor for constexpr initialization with hardcoded BOP candidates
    constexpr BOPCandidateOffsets() {
        // Initialize with standard BOP candidate offsets
        // {1, -1, 2, -2, 3, -3, ..., 40, -40}
        const int init_vals[BOP_NUM_CANDIDATES] = BOP_CANDIDATES_INIT;
        for (int i = 0; i < BOP_NUM_CANDIDATES; i++) {
            values[i] = init_vals[i];
        }
    }
};

// ============================================================================
// Initialization Functions (Constexpr)
// ============================================================================

// ========================================================================
// initBOPRecencyRing: Create initial recency ring matrix
// ========================================================================
inline constexpr BOPRecencyRingMatrix initBOPRecencyRing() {
    return BOPRecencyRingMatrix();
}

// ========================================================================
// initBOPPatternLearner: Create initial pattern learner matrix
// ========================================================================
inline constexpr BOPPatternLearnerMatrix initBOPPatternLearner() {
    return BOPPatternLearnerMatrix();
}

// ========================================================================
// initBOPPrefetchBuffer: Create initial prefetch buffer matrix
// ========================================================================
inline constexpr BOPPrefetchBufferMatrix initBOPPrefetchBuffer() {
    return BOPPrefetchBufferMatrix();
}

// ========================================================================
// initBOPCandidateOffsets: Create initial candidate offsets
// ========================================================================
inline constexpr BOPCandidateOffsets initBOPCandidateOffsets() {
    return BOPCandidateOffsets();
}

// ============================================================================
// Type Aliases for Convenience
// ============================================================================
using BOP = class BOPrefetcher;  // Forward declaration for typedef
