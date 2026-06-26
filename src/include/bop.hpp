#pragma once

#include "bop_config.hpp"
#include "bop_data_type.hpp"
#include "bop_recency_ring.hpp"
#include "bop_pattern_learner.hpp"
#include "bop_prefetch_buffer.hpp"
#include "bop_init.hpp"

// ============================================================================
// BOP Prefetcher - Main Class
// ============================================================================
// Best Offset Pattern Prefetcher: Tests various cache-line offsets to discover
// recurring patterns in access streams. Maintains a recency ring of recently
// accessed addresses and scores candidate offsets based on pattern matches.
//
// Algorithm Overview:
// 1. On each cache access, test a candidate offset against the recency ring
// 2. If (address - offset) is in RR, increment the offset's score
// 3. Insert current address into recency ring
// 4. Periodically check if learning phase should end (max rounds or max score)
// 5. When phase ends, select top-N offsets and begin new learning phase
// 6. Issue prefetches based on best offsets
// 7. On cache fills, insert backward-predicted addresses into RR
//
// ============================================================================
// Compilation Modes (controlled by BOP_SINGLE_PREFETCH in bop_config.hpp)
// ============================================================================
// BOP_SINGLE_PREFETCH = 1 (Default for HLS):
//   - Issues exactly 1 prefetch per cycle (II=1)
//   - Uses only the best-scoring offset
//   - Optimized for single-cycle throughput in hardware
//   - Minimal resource usage, deterministic latency
//   - Code path: Compiled with #if BOP_SINGLE_PREFETCH
//
// BOP_SINGLE_PREFETCH = 0 (For multi-prefetch operation):
//   - Issues up to BOP_TOP_N prefetches per cycle
//   - Uses variable-length loop over best offsets
//   - Higher throughput but may require II > 1
//   - More resource usage in hardware
//   - Code path: Compiled with #else (when BOP_SINGLE_PREFETCH = 0)
//
// To switch modes, edit bop_config.hpp:
//   #define BOP_SINGLE_PREFETCH 1   // For single prefetch (HLS optimized)
//   #define BOP_SINGLE_PREFETCH 0   // For multiple prefetches
//
// Template Parameters (with default types from bop_data_type.hpp):
//   address_t: Full memory address type
//   block_address_t: Cache-line address type
//   rr_entry_t: Recency ring entry type
//   candidate_t: Candidate offset type
//   score_t: Score counter type
//   candidate_index_t: Index into candidates array

template <typename address_t = bop_address_t,
          typename block_address_t = bop_block_address_t,
          typename rr_entry_t = bop_rr_entry_t,
          typename candidate_t = bop_candidate_t,
          typename score_t = bop_score_t,
          typename candidate_index_t = bop_candidate_index_t>
class BOPrefetcher {
public: // <-- Asegurado public para poder aplicar pragmas a los miembros
    // Component instances
    BOPRecencyRing<rr_entry_t> recency_ring;
    BOPPatternLearner<candidate_t, score_t, candidate_index_t> pattern_learner;
    BOPPrefetchBuffer<address_t> prefetch_buffer;

    // Constructor: trivial initialization
    BOPrefetcher() = default;

    // ========================================================================
    // process_cache_access: Main prefetcher logic - called on every cache hit/miss
    // ========================================================================
    // Signature: Same as GASP/SPP pattern for consistency
    //   addr: Full memory address of access
    //   prefetch_deltas: Output array for cache-line offsets to prefetch
    //   prefetch_confidences: Output array for confidence scores (unused for BOP)
    //   num_prefetches: Output count of prefetches to issue
    void process_cache_access(address_t addr,
                             bop_offset_t* prefetch_deltas,
                             score_t* prefetch_confidences,
                             uint32_t& num_prefetches) {
        #pragma HLS PIPELINE

        // ====================================================================
        // HW Object Array Partitioning
        // ====================================================================
        // Force array partitioning on the ACTUAL hardware instances, not just
        // the static initialization matrices, to prevent II violations during reset.
        #pragma HLS ARRAY_PARTITION variable=pattern_learner.scores complete
        #pragma HLS ARRAY_PARTITION variable=pattern_learner.best_offsets complete

        // ====================================================================
        // Static Initialization (Constexpr - Compile-time)
        // ====================================================================
        // Initialize BOP structures once using constexpr functions
        static const BOPRecencyRingMatrix<bop_rr_entry_t, bop_rr_index_t> rr_matrix = initBOPRecencyRing<bop_rr_entry_t, bop_rr_index_t>();

        static const BOPPatternLearnerMatrix<bop_score_t, bop_candidate_loop_t, bop_counter_t, bop_candidate_t> pl_matrix =
                initBOPPatternLearner<bop_score_t, bop_candidate_loop_t, bop_counter_t, bop_candidate_t>();

        static const BOPPrefetchBufferMatrix<bop_address_t, bop_rr_index_t> pb_matrix = initBOPPrefetchBuffer<bop_address_t, bop_rr_index_t>();

        static const BOPCandidateOffsets<bop_candidate_t> candidates = initBOPCandidateOffsets<bop_candidate_t>();
        #pragma HLS ARRAY_PARTITION variable=candidates.values complete

        // ====================================================================
        // Extract block address for pattern matching
        // ====================================================================
        block_address_t block_addr = (addr >> BOP_LOG2_BLOCK_SIZE);

        // Extract page and offset for prefetch generation
        bop_page_t page = (addr >> BOP_LOG2_PAGE_SIZE);
        bop_offset_t page_offset = ((addr >> BOP_LOG2_BLOCK_SIZE) & BOP_PAGE_OFFSET_MASK);

        // ====================================================================
        // Stage 1: Test Current Candidate Offset
        // ====================================================================
        // Check if (block_addr - candidate) is in the recency ring
        candidate_t test_offset = candidates.values[pattern_learner.candidate_ptr];
        block_address_t test_addr = block_addr - test_offset;

        if (recency_ring.search_entry(test_addr)) {
            // Pattern matched: increment score for this offset
            pattern_learner.increment_score();
        }

        // ====================================================================
        // Stage 2: Insert Current Address into Recency Ring
        // ====================================================================
        recency_ring.insert_entry(block_addr);

        // ====================================================================
        // Stage 3 & 4: Phase Transition and Advance (Mutually Exclusive)
        // ====================================================================
        // We check phase_end first. If the phase ends, we reset (which resets
        // the candidate_ptr to 0). If it hasn't ended, we move to the next.
        // This entirely eliminates the double-write (WAW) dependency.
        bop_valid_t phase_end = pattern_learner.check_phase_end();

        if (phase_end) {
            // Phase ends: select best offsets and reset scores
            candidate_index_t best_indices[BOP_TOP_N];
            pattern_learner.select_best_offsets_indices(best_indices);

            // Map indices to actual offset values
            for (bop_top_n_index_t i = 0; i < BOP_TOP_N; i++) {
#pragma HLS UNROLL
                pattern_learner.best_offsets[i] = candidates.values[best_indices[i]];
            }

            // Reset for next learning phase
            pattern_learner.reset_phase();
        } else {
            // Only advance if phase has not ended
            pattern_learner.next_candidate();
        }

        // ====================================================================
        // Stage 5: Prefetch Generation (Single Prefetch Per Cycle)
        // ====================================================================
        // Generate prefetches based on best offsets discovered so far
        num_prefetches = 0;

#if BOP_SINGLE_PREFETCH
        // ====================================================================
        // Optimized Mode: Single Prefetch Per Cycle (II=1)
        // ====================================================================
        // Use only the best offset, eliminating variable-length loop
        // This path is compiled-in when BOP_SINGLE_PREFETCH=1
        if (pattern_learner.num_best_offsets > 0) {
            bop_offset_t pf_offset = pattern_learner.best_offsets[0];
            bop_offset_t final_offset = page_offset + pf_offset;

            // Check bounds: offset must be within page
            if (final_offset >= 0 && final_offset < BOP_PAGE_OFFSET_MASK + 1) {
                prefetch_deltas[0] = final_offset - page_offset;  // Delta from current
                prefetch_confidences[0] = pattern_learner.scores[0];  // Use score as confidence
                num_prefetches = 1;
            }
        }

#else
        // ====================================================================
        // Full Mode: Multiple Prefetches (Variable Prefetch Degree)
        // ====================================================================
        // Issue up to BOP_TOP_N prefetches per cycle
        // This path is compiled-in when BOP_SINGLE_PREFETCH=0
        #pragma HLS UNROLL FACTOR=2
        for (bop_top_n_index_t i = 0; i < BOP_TOP_N; i++) {
            #pragma HLS UNROLL
            if (i < pattern_learner.num_best_offsets && num_prefetches < BOP_PREF_DEGREE) {
                bop_offset_t pf_offset = pattern_learner.best_offsets[i];
                bop_offset_t final_offset = page_offset + pf_offset;

                if (final_offset >= 0 && final_offset < BOP_PAGE_OFFSET_MASK + 1) {
                    prefetch_deltas[num_prefetches] = final_offset - page_offset;
                    prefetch_confidences[num_prefetches] = pattern_learner.scores[i];
                    num_prefetches++;
                }
            }
        }

#endif
    }

    // ========================================================================
    // notify_cache_fill: Called when a cache fill completes
    // ========================================================================
    // Insert backward-predicted addresses into recency ring
    void notify_cache_fill(address_t filled_addr) {
        /*
        #pragma HLS PIPELINE II=1

        static const BOPCandidateOffsets candidates = initBOPCandidateOffsets();
        #pragma HLS ARRAY_PARTITION variable=candidates.values complete

        block_address_t block_addr = (filled_addr >> BOP_LOG2_BLOCK_SIZE);

        // For each known good offset, insert backward predictions into RR
        #pragma HLS UNROLL FACTOR=2
        for (bop_top_n_index_t i = 0; i < BOP_TOP_N; i++) {
            #pragma HLS UNROLL
            if (i < pattern_learner.num_best_offsets) {
                candidate_t offset = pattern_learner.best_offsets[i];
                block_address_t pred_addr = block_addr - offset;
                recency_ring.insert_entry(pred_addr);
            }
        }
        */
    }

    // ========================================================================
    // clear: Reset prefetcher state
    // ========================================================================
    void clear() {
        #pragma HLS PIPELINE II=1

        recency_ring.clear();
        pattern_learner.clear();
        prefetch_buffer.clear();
    }
};
