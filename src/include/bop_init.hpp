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
template<typename bop_rr_entry_t, typename bop_rr_index_t>
struct BOPRecencyRingMatrix {
    bop_rr_entry_t entries[BOP_RR_SIZE];
    bop_rr_index_t head;
    // Eliminado: BOPRecencyRingMatrix(){};
};

// ============================================================================
// Pattern Learner Matrix Structure
// ============================================================================
template<typename bop_score_t, typename bop_candidate_loop_t, typename bop_counter_t, typename bop_candidate_t>
struct BOPPatternLearnerMatrix {
    bop_score_t scores[BOP_NUM_CANDIDATES];
    bop_candidate_loop_t candidate_ptr;
    bop_counter_t round_counter;
    bop_candidate_t best_offsets[BOP_TOP_N];
    uint32_t num_best_offsets;
    // Eliminado: BOPPatternLearnerMatrix(){};
};

// ============================================================================
// Prefetch Buffer Matrix Structure
// ============================================================================
template<typename bop_address_t, typename bop_rr_index_t>
struct BOPPrefetchBufferMatrix {
    bop_address_t buffer[BOP_PREF_BUFFER_SIZE];
    bop_rr_index_t head;
    bop_rr_index_t tail;
    bop_rr_index_t count;
    // Eliminado: BOPPrefetchBufferMatrix(){};
};

// ============================================================================
// Candidate Offsets Storage
// ============================================================================
template<typename bop_candidate_t>
struct BOPCandidateOffsets {
    bop_candidate_t values[BOP_NUM_CANDIDATES];
    // Eliminado: BOPCandidateOffsets(){};
};

// ============================================================================
// Initialization Functions (Constexpr)
// ============================================================================

// ========================================================================
// initBOPRecencyRing: Create initial recency ring matrix
// ========================================================================
template<typename bop_rr_entry_t, typename bop_rr_index_t>
constexpr BOPRecencyRingMatrix<bop_rr_entry_t, bop_rr_index_t> initBOPRecencyRing() {
    BOPRecencyRingMatrix<bop_rr_entry_t, bop_rr_index_t> res;
    res.head = 0;
    for(int i = 0; i < BOP_RR_SIZE; i++) {
        res.entries[i] = 0;
    }
    return res;
}

// ========================================================================
// initBOPPatternLearner: Create initial pattern learner matrix
// ========================================================================
template<typename bop_score_t, typename bop_candidate_loop_t, typename bop_counter_t, typename bop_candidate_t>
constexpr BOPPatternLearnerMatrix<bop_score_t, bop_candidate_loop_t, bop_counter_t, bop_candidate_t> initBOPPatternLearner() {
    BOPPatternLearnerMatrix<bop_score_t, bop_candidate_loop_t, bop_counter_t, bop_candidate_t> res;
    res.candidate_ptr = 0;
    res.round_counter = 0;
    res.num_best_offsets = 0;
    for (int i = 0; i < BOP_NUM_CANDIDATES; i++) {
        res.scores[i] = 0;
    }
    for (int i = 0; i < BOP_TOP_N; i++) {
        res.best_offsets[i] = 0;
    }
    return res;
}

// ========================================================================
// initBOPPrefetchBuffer: Create initial prefetch buffer matrix
// ========================================================================
template<typename bop_address_t, typename bop_rr_index_t>
constexpr BOPPrefetchBufferMatrix<bop_address_t, bop_rr_index_t> initBOPPrefetchBuffer() {
    BOPPrefetchBufferMatrix<bop_address_t, bop_rr_index_t> res;
    res.head = 0;
    res.tail = 0;
    res.count = 0;
    for (int i = 0; i < BOP_PREF_BUFFER_SIZE; i++) {
        res.buffer[i] = 0;
    }
    return res;
}

// ========================================================================
// initBOPCandidateOffsets: Create initial candidate offsets
// ========================================================================
static const int candidates[BOP_NUM_CANDIDATES] = BOP_CANDIDATES_INIT;
template<typename bop_candidate_t>
constexpr BOPCandidateOffsets<bop_candidate_t> initBOPCandidateOffsets() {
    BOPCandidateOffsets<bop_candidate_t> res;
    for (int i = 0; i < BOP_NUM_CANDIDATES; i++) {
            res.values[i] = candidates[i];
        }
    return res;
}

// ============================================================================
// Type Aliases for Convenience
// ============================================================================
// using BOP = class BOPrefetcher;  // Forward declaration for typedef
