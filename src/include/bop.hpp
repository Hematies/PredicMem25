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
private:
    // Component instances
    BOPRecencyRing<rr_entry_t> recency_ring;
    BOPPatternLearner<candidate_t, score_t, candidate_index_t> pattern_learner;
    BOPPrefetchBuffer<address_t> prefetch_buffer;

public:
    // Constructor: trivial initialization
    BOPrefetcher() = default;

    // ========================================================================
    // process_cache_access: Main prefetcher logic - called on every cache hit/miss
    // ========================================================================
    // Signature: Same as GASP/SPP pattern for consistency
    //   addr: Full memory address of access
    //   cache_hit: Whether this access hit in cache (1=hit, 0=miss)
    //   useful_prefetch: Whether any prefetch was useful (for feedback)
    //   prefetch_deltas: Output array for cache-line offsets to prefetch
    //   prefetch_confidences: Output array for confidence scores (unused for BOP)
    //   num_prefetches: Output count of prefetches to issue
    //   num_prefetches_l2: Output count for L2 (unused)
    void process_cache_access(address_t addr,
                             bop_valid_t cache_hit,
                             bop_valid_t useful_prefetch,
                             bop_offset_t* prefetch_deltas,
                             score_t* prefetch_confidences,
                             uint32_t& num_prefetches,
                             uint32_t& num_prefetches_l2) {
        #pragma HLS PIPELINE II=1

        // ====================================================================
        // Static Initialization (Constexpr - Compile-time)
        // ====================================================================
        // Initialize BOP structures once using constexpr functions
        static const BOPRecencyRingMatrix rr_matrix = initBOPRecencyRing();
        #pragma HLS ARRAY_RESHAPE variable=rr_matrix.entries cyclic=16

        static const BOPPatternLearnerMatrix pl_matrix = initBOPPatternLearner();
        #pragma HLS ARRAY_PARTITION variable=pl_matrix.scores complete

        static const BOPPrefetchBufferMatrix pb_matrix = initBOPPrefetchBuffer();
        #pragma HLS ARRAY_PARTITION variable=pb_matrix.buffer cyclic=4

        static const BOPCandidateOffsets candidates = initBOPCandidateOffsets();
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
        // Stage 3: Advance to Next Candidate
        // ====================================================================
        pattern_learner.next_candidate();

        // ====================================================================
        // Stage 4: Check for Phase Transition
        // ====================================================================
        bop_valid_t phase_end = pattern_learner.check_phase_end();

        if (phase_end) {
            // Phase ends: select best offsets and reset scores
            candidate_index_t best_indices[BOP_TOP_N];
            pattern_learner.select_best_offsets_indices(best_indices);

            // Map indices to actual offset values
            #pragma HLS UNROLL
            for (bop_top_n_index_t i = 0; i < BOP_TOP_N; i++) {
                #pragma HLS UNROLL
                pattern_learner.best_offsets[i] = candidates.values[best_indices[i]];
            }

            // Reset for next learning phase
            pattern_learner.reset_phase();
        }

        // ====================================================================
        // Stage 5: Prefetch Generation (Single Prefetch Per Cycle)
        // ====================================================================
        // Generate prefetches based on best offsets discovered so far
        num_prefetches = 0;
        num_prefetches_l2 = 0;

        // For single prefetch per cycle: use first (best) offset only
        if (pattern_learner.num_best_offsets > 0 && BOP_SINGLE_PREFETCH) {
            bop_offset_t pf_offset = pattern_learner.best_offsets[0];
            bop_offset_t final_offset = page_offset + pf_offset;

            // Check bounds: offset must be within page
            if (final_offset >= 0 && final_offset < BOP_PAGE_OFFSET_MASK + 1) {
                prefetch_deltas[0] = final_offset - page_offset;  // Delta from current
                prefetch_confidences[0] = pattern_learner.scores[0];  // Use score as confidence
                num_prefetches = 1;
            }
        } else if (!BOP_SINGLE_PREFETCH) {
            // Multiple prefetches mode (not optimized for HLS)
            #pragma HLS UNROLL FACTOR=2
            for (bop_top_n_index_t i = 0; i < BOP_TOP_N; i++) {
                #pragma HLS UNROLL
                if (i < pattern_learner.num_best_offsets) {
                    bop_offset_t pf_offset = pattern_learner.best_offsets[i];
                    bop_offset_t final_offset = page_offset + pf_offset;

                    if (final_offset >= 0 && final_offset < BOP_PAGE_OFFSET_MASK + 1) {
                        prefetch_deltas[num_prefetches] = final_offset - page_offset;
                        prefetch_confidences[num_prefetches] = pattern_learner.scores[i];
                        num_prefetches++;
                    }
                }
            }
        }
    }

    // ========================================================================
    // notify_cache_fill: Called when a cache fill completes
    // ========================================================================
    // Insert backward-predicted addresses into recency ring
    void notify_cache_fill(address_t filled_addr) {
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
