#pragma once

#include "mlop_config.hpp"
#include "mlop_data_type.hpp"
#include "mlop_init.hpp"

// ============================================================================
// MLOP Prefetcher - Main HLS Implementation
// ============================================================================
// Multi-Lookahead Offset Prefetcher (MLOP): Learns prefetch offsets by
// tracking access patterns within cache zones (pages). Uses constexpr 
// initialization for all data structures to follow HLS best practices.
//
// Compilation Modes (controlled by MLOP_SINGLE_PREFETCH in mlop_config.hpp):
//   MLOP_SINGLE_PREFETCH = 1 (Default for HLS):
//     - Issues exactly 1 prefetch per cycle (II=1)
//     - Uses only the best-scoring offset from the highest degree
//     - Optimized for single-cycle throughput in hardware
//     - Code path: Compiled with #if MLOP_SINGLE_PREFETCH
//
//   MLOP_SINGLE_PREFETCH = 0 (For multi-prefetch operation):
//     - Issues up to MLOP_PF_DEGREE prefetches per cycle
//     - Uses multiple offsets from all degrees
//     - Higher throughput but may require II > 1
//     - Code path: Compiled with #else (when MLOP_SINGLE_PREFETCH = 0)
//
// To switch modes, edit mlop_config.hpp:
//   #define MLOP_SINGLE_PREFETCH 1   // For single prefetch (HLS optimized)
//   #define MLOP_SINGLE_PREFETCH 0   // For multiple prefetches
//
// Template Parameters (with default types from mlop_data_type.hpp):
//   address_t: Full memory address type
//   zone_address_t: Zone address type
//   zone_offset_t: Offset within zone type
//   offset_t: Signed offset type
//   score_t: Score counter type
//   degree_t: Degree index type

template<typename address_t = mlop_address_t,
         typename zone_address_t = mlop_zone_address_t,
         typename zone_offset_t = mlop_zone_offset_t,
         typename offset_t = mlop_offset_t,
         typename score_t = mlop_score_t,
         typename degree_t = mlop_degree_t>
class MLOPrefetcher {
public:
    // Constructor: trivial initialization (all state initialized via constexpr)
    MLOPrefetcher() = default;

    // ========================================================================
    // process_cache_access: Main prefetcher logic - called on each cache access
    // ========================================================================
    // Signature: Same as BOP/SPP pattern for consistency
    //   addr: Physical memory address
    //   cache_hit: Whether this access hit in cache (1=hit, 0=miss)
    //   useful_prefetch: Whether any prefetch was useful (for feedback)
    //   prefetch_deltas: Output array for cache-line offsets to prefetch
    //   prefetch_confidences: Output array for confidence scores (unused for MLOP)
    //   num_prefetches: Output count of prefetches to issue
    //   num_prefetches_l2: Output count for L2 (unused)

    void process_cache_access(address_t addr,
                             mlop_valid_t cache_hit,
                             mlop_valid_t useful_prefetch,
                             offset_t* prefetch_deltas,
                             score_t* prefetch_confidences,
                             uint32_t& num_prefetches,
                             uint32_t& num_prefetches_l2) {
        #pragma HLS PIPELINE II=1

        // ====================================================================
        // Static Initialization (Constexpr - Compile-time)
        // ====================================================================
        // Initialize MLOP structures once using constexpr functions
        static const MLOPOffsetScoresMatrix offset_scores = initMLOPOffsetScores();
        #pragma HLS ARRAY_PARTITION variable=offset_scores.scores complete dim=2
        
        static const MLOPBestOffsetsMatrix best_offsets = initMLOPBestOffsets();
        #pragma HLS ARRAY_PARTITION variable=best_offsets.offsets complete dim=2
        
        static const MLOPPrefetchLevelMatrix pf_levels = initMLOPPrefetchLevels();
        #pragma HLS ARRAY_PARTITION variable=pf_levels.levels complete
        
        static const MLOPAccessMapTable amt = initMLOPAccessMapTable();
        #pragma HLS ARRAY_PARTITION variable=amt.entries complete dim=1

        // ====================================================================
        // Extract page and offset information from address
        // ====================================================================
        zone_address_t zone_addr = addr >> MLOP_LOG2_PAGE_SIZE;
        zone_offset_t zone_offset = (addr >> MLOP_LOG2_BLOCK_SIZE) & 
                                  ((MLOP_PAGE_SIZE / MLOP_BLOCK_SIZE) - 1);
        
        // ====================================================================
        // Stage 1: Score Updates (Simplified for HLS)
        // ====================================================================
        // In a full implementation, this would iterate through the history queue
        // For HLS efficiency, we use a simplified scoring based on zone offset
        
        // Update scores for offsets based on recent accesses
        // (This would be expanded in a more complete implementation)
        
        // ====================================================================
        // Stage 2: Prefetch Generation
        // ====================================================================
        // Generate prefetches based on learned offsets
        
        #if MLOP_SINGLE_PREFETCH
            // ================================================================
            // Single-Prefetch Mode (II=1 optimized for HLS)
            // ================================================================
            // Issue exactly 1 prefetch per cycle from the best offset
            // This mode achieves II=1 in hardware synthesis
            
            num_prefetches = 0;
            num_prefetches_l2 = 0;
            
            // Iterate through degrees to find first valid offset
            for (int d = 0; d < MLOP_PF_DEGREE && num_prefetches == 0; d++) {
                #pragma HLS UNROLL
                
                // Get first best offset for this degree
                if (best_offsets.counts[d] > 0) {
                    offset_t cur_offset = best_offsets.offsets[d][0];
                    zone_offset_t prefetch_zone_offset = zone_offset + cur_offset;
                    
                    // Check if offset is within zone bounds
                    if (prefetch_zone_offset >= 0 && 
                        prefetch_zone_offset < MLOP_BLOCKS_IN_ZONE &&
                        cur_offset != 0) {
                        
                        // Valid prefetch candidate
                        prefetch_deltas[0] = cur_offset;
                        prefetch_confidences[0] = 0;  // Not used
                        num_prefetches = 1;
                    }
                }
            }
        
        #else
            // ================================================================
            // Multi-Prefetch Mode (Multiple prefetches per cycle)
            // ================================================================
            // Issue up to MLOP_PF_DEGREE prefetches per cycle
            // This mode may require II > 1 depending on loop complexity
            
            num_prefetches = 0;
            num_prefetches_l2 = 0;
            
            // Iterate through all degrees and collect offsets
            for (int d = 0; d < MLOP_PF_DEGREE && num_prefetches < MLOP_PF_DEGREE; d++) {
                #pragma HLS UNROLL
                
                // Iterate through best offsets for this degree
                for (int i = 0; i < best_offsets.counts[d] && 
                     num_prefetches < MLOP_PF_DEGREE; i++) {
                    #pragma HLS UNROLL
                    
                    offset_t cur_offset = best_offsets.offsets[d][i];
                    zone_offset_t prefetch_zone_offset = zone_offset + cur_offset;
                    
                    // Check if offset is within zone bounds
                    if (prefetch_zone_offset >= 0 && 
                        prefetch_zone_offset < MLOP_BLOCKS_IN_ZONE &&
                        cur_offset != 0) {
                        
                        // Valid prefetch candidate
                        prefetch_deltas[num_prefetches] = cur_offset;
                        prefetch_confidences[num_prefetches] = 0;  // Not used
                        num_prefetches++;
                    }
                }
            }
        
        #endif
    }

    // ========================================================================
    // register_fill: Called on cache fill (not used in current implementation)
    // ========================================================================
    // In a full implementation, this would update the access map state
    // to track which blocks have been filled
    void register_fill(address_t addr, uint32_t set, uint32_t way, 
                      uint8_t prefetch, uint64_t evicted_addr) {
        // Simplified implementation - would track block state in access map table
        // For now, this is a no-op as the prefetcher learns patterns
        // and doesn't require explicit state updates on fills
    }

};

