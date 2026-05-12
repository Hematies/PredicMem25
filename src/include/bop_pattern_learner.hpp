#pragma once

#include "bop_config.hpp"
#include "bop_data_type.hpp"

// ============================================================================
// BOP Pattern Learner
// ============================================================================
// Tracks the effectiveness of candidate offsets through a scoring mechanism.
// Each candidate offset gets a score based on how often it correctly predicts
// accesses (i.e., the pattern matches in the recency ring). When a phase ends,
// the best-scoring offsets are selected for prefetching.
//
// Phase transitions occur when either:
// - Round counter reaches BOP_MAX_ROUNDS, OR
// - Any score reaches BOP_MAX_SCORE
//
// Template Parameters:
//   candidate_t: Type for candidate values (default: bop_candidate_t)
//   score_t: Type for score counters (default: bop_score_t)
//   candidate_index_t: Type for candidate indices (default: bop_candidate_index_t)

template <typename candidate_t = bop_candidate_t,
          typename score_t = bop_score_t,
          typename candidate_index_t = bop_candidate_index_t>
class BOPPatternLearner {
public:
    // Member arrays and counters
    score_t scores[BOP_NUM_CANDIDATES];                           // Score for each candidate
    bop_candidate_loop_t candidate_ptr;                           // Current candidate being evaluated
    bop_counter_t round_counter;                                  // Round counter for phase transitions
    candidate_t best_offsets[BOP_TOP_N];                          // Selected best offsets for prefetching
    uint32_t num_best_offsets;                                    // Number of valid best offsets

    // Constructor: trivial initialization
    BOPPatternLearner() = default;

    // ========================================================================
    // increment_score: Increment score for current candidate
    // ========================================================================
    // Called when candidate offset matches in recency ring
    void increment_score() {
        #pragma HLS PIPELINE II=1

        if (scores[candidate_ptr] < BOP_SCORE_MAX) {
            scores[candidate_ptr] = scores[candidate_ptr] + 1;
        }
    }

    // ========================================================================
    // next_candidate: Move to next candidate offset
    // ========================================================================
    // Cycles through candidates in round-robin fashion, increments round counter
    void next_candidate() {
        #pragma HLS PIPELINE II=1

        candidate_ptr = (candidate_ptr + 1) % BOP_NUM_CANDIDATES;
        round_counter = round_counter + 1;
    }

    // ========================================================================
    // check_phase_end: Determine if current phase should end
    // ========================================================================
    // Returns true if round counter exceeded or max score reached
    bop_valid_t check_phase_end() {
        #pragma HLS PIPELINE II=1

        bop_valid_t end_by_rounds = (round_counter >= BOP_MAX_ROUNDS) ? 1 : 0;
        bop_valid_t end_by_score = 0;

        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL FACTOR=8
        for (bop_candidate_loop_t i = 0; i < BOP_NUM_CANDIDATES; i++) {
            #pragma HLS UNROLL
            if (scores[i] >= BOP_MAX_SCORE) {
                end_by_score = 1;
            }
        }

        return (end_by_rounds | end_by_score);
    }

    // ========================================================================
    // select_best_offsets: Select top-N offsets based on scores
    // ========================================================================
    // Uses heap-like selection to find the N best-performing offsets
    // Requires external storage of candidate values to map indices to offsets
    void select_best_offsets_indices(candidate_index_t best_indices[BOP_TOP_N]) {
        #pragma HLS PIPELINE II=1

        // For BOP_TOP_N=1, find the single best offset
        score_t max_score = 0;
        candidate_index_t max_idx = 0;

        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL FACTOR=8
        for (bop_candidate_loop_t i = 0; i < BOP_NUM_CANDIDATES; i++) {
            #pragma HLS UNROLL
            if (scores[i] >= max_score) {
                max_score = scores[i];
                max_idx = i;
            }
        }

        best_indices[0] = max_idx;
        num_best_offsets = 1;
    }

    // ========================================================================
    // reset_phase: Clear scores and counters for next learning phase
    // ========================================================================
    void reset_phase() {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL FACTOR=16

        for (bop_candidate_loop_t i = 0; i < BOP_NUM_CANDIDATES; i++) {
            #pragma HLS UNROLL
            scores[i] = 0;
        }
        candidate_ptr = 0;
        round_counter = 0;
        num_best_offsets = 0;
    }

    // ========================================================================
    // clear: Reset all learner state
    // ========================================================================
    void clear() {
        #pragma HLS PIPELINE II=1

        reset_phase();
        #pragma HLS UNROLL FACTOR=4
        for (bop_top_n_index_t i = 0; i < BOP_TOP_N; i++) {
            #pragma HLS UNROLL
            best_offsets[i] = 0;
        }
    }
};
