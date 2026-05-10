#pragma once

#include "spp_config.hpp"
#include "spp_data_type.hpp"
#include "spp_signature_table.hpp"
#include "spp_pattern_table.hpp"
#include "spp_prefetch_filter.hpp"
#include "spp_global_register.hpp"

// ============================================================================
// SPP (Signature Path Prefetcher) - Main HLS Implementation
// ============================================================================
// Port of ChampSim SPP to HLS for hardware implementation
//
// Template parameters define the data types and bitwidths for:
// - Addresses and cache lines
// - Signatures and deltas
// - Confidence values and counters
// - Prefetch queue management

template<typename address_t = spp_address_t, 
         typename block_address_t = spp_block_address_t, 
         typename st_tag_t = spp_st_tag_t, 
         typename st_sig_t = spp_st_sig_t, 
         typename st_confidence_t = spp_st_confidence_t, 
         typename pt_delta_t = spp_pt_delta_t, 
         typename pt_confidence_t = spp_pt_confidence_t, 
         typename filter_tag_t = spp_filter_tag_t>
class SPP {
public:
    // Core SPP structures
    SPPSignatureTable<st_tag_t, st_sig_t, st_confidence_t> signature_table;
    SPPPatternTable<pt_delta_t, pt_confidence_t> pattern_table;
    SPPPrefetchFilter<filter_tag_t> prefetch_filter;
    SPPGlobalRegister<st_tag_t, st_sig_t, st_confidence_t, pt_delta_t> global_register;

    // Constructor
    SPP() {
        // Structures initialize themselves
    }

    // ========================================================================
    // Main prefetcher interface
    // ========================================================================
    // Processes a cache access and generates a single prefetch request per cycle
    //
    // Input:
    //   - addr: Physical memory address
    //   - cache_hit: Whether this was a cache hit or miss
    //   - useful_prefetch: Whether previous prefetch was useful (feedback)
    //
    // Output:
    //   - prefetch_deltas: Single prefetch delta (highest confidence)
    //   - prefetch_confidences: Confidence score for the prefetch
    //   - num_prefetches: 0 or 1 (prefetch valid or not)
    //   - num_prefetches_l2: 0 or 1 (prefetch tier level)

    void process_cache_access(address_t addr,
                             spp_ghr_valid_t cache_hit,
                             spp_ghr_valid_t useful_prefetch,
                             pt_delta_t* prefetch_deltas,
                             pt_confidence_t* prefetch_confidences,
                             uint32_t& num_prefetches,
                             uint32_t& num_prefetches_l2) {
        #pragma HLS PIPELINE II=1
        
        // Extract page and offset information
        spp_address_t page = addr >> SPP_LOG2_PAGE_SIZE;
        spp_page_offset_t page_offset = (addr >> SPP_LOG2_BLOCK_SIZE) & 
                                  ((SPP_PAGE_SIZE / SPP_BLOCK_SIZE) - 1);
        
        st_sig_t last_sig = 0;
        st_sig_t curr_sig = 0;
        pt_delta_t delta = 0;

        // ====================================================================
        // Stage 1: Signature Table Lookup and Update
        // ====================================================================
        // Read current signature and calculate delta from last access
        signature_table.read_and_update_sig(page, page_offset, 
                                           last_sig, curr_sig, delta);

        // ====================================================================
        // Stage 2: Pattern Table Update
        // ====================================================================
        // Update pattern correlation if we have a previous signature
        if (last_sig != 0) {
            pattern_table.update_pattern(last_sig, delta);
        }

        // ====================================================================
        // Stage 3: Global Accuracy Update
        // ====================================================================
        global_register.update_global_accuracy();

        // ====================================================================
        // Stage 4: Prefetch Generation (Single Prefetch Per Cycle)
        // ====================================================================
        // Generate only the highest-confidence prefetch
        block_address_t base_addr = addr & ~(SPP_BLOCK_SIZE - 1);
        
        num_prefetches = 0;
        num_prefetches_l2 = 0;

        // Read patterns for current signature and find max confidence prefetch
        spp_pt_set_index_t pt_set = SPPPatternTable<pt_delta_t, 
                                                    pt_confidence_t>::hash_signature(curr_sig) % 
                                    SPP_PT_SET;
        
        pt_delta_t best_delta = 0;
        pt_confidence_t best_conf = 0;
        spp_pt_way_index_t best_way = SPP_PT_WAY;

        // Find highest confidence delta in pattern table
        #pragma HLS UNROLL
        for (spp_pt_way_index_t way = 0; way < SPP_PT_WAY; way++) {
            pt_confidence_t local_conf = (pattern_table.c_sig[pt_set] > 0) ?
                (100 * pattern_table.c_delta[pt_set][way]) / pattern_table.c_sig[pt_set] : 0;
            
            if (local_conf > best_conf && local_conf >= SPP_PF_THRESHOLD) {
                best_conf = local_conf;
                best_delta = pattern_table.delta[pt_set][way];
                best_way = way;
            }
        }

        // If a valid prefetch was found
        if (best_way < SPP_PT_WAY && best_conf >= SPP_PF_THRESHOLD) {
            // Calculate prefetch address
            block_address_t pf_addr = base_addr + 
                                      (best_delta << SPP_LOG2_BLOCK_SIZE);

            // Check if within same page (page boundary protection)
            if ((addr & ~(SPP_PAGE_SIZE - 1)) == 
                (pf_addr & ~(SPP_PAGE_SIZE - 1))) {
                
                // Determine L2 vs LLC prefetch based on confidence
                SPPFilterRequest request_type = 
                    (best_conf >= SPP_FILL_THRESHOLD) ? 
                    SPP_L2_PREFETCH : SPP_LLC_PREFETCH;

                // Check filter and issue if allowed
                spp_ghr_valid_t should_prefetch = 
                    prefetch_filter.check(pf_addr, request_type,
                                        global_register.pf_issued,
                                        global_register.pf_useful);

                if (should_prefetch) {
                    prefetch_deltas[0] = best_delta;
                    prefetch_confidences[0] = best_conf;
                    
                    num_prefetches = 1;
                    
                    if (request_type == SPP_L2_PREFETCH) {
                        num_prefetches_l2 = 1;
                        // Increment prefetch issued counter with saturation
                        global_register.increment_pf_issued();
                        if (global_register.pf_issued > SPP_GLOBAL_COUNTER_MAX) {
                            global_register.pf_issued >>= 1;
                            global_register.pf_useful >>= 1;
                        }
                    }
                }
            } else {
                // Cross-page prefetch - store in GHR for future learning
                if constexpr (SPP_GHR_ON) {
                    spp_ghr_offset_t pf_offset = (pf_addr >> SPP_LOG2_BLOCK_SIZE) & 0x3F;
                    global_register.update_entry(curr_sig, 
                                                best_conf,
                                                pf_offset,
                                                best_delta);
                }
            }
        }
    }

    // ========================================================================
    // Cache fill notification
    // ========================================================================
    // Called when a cache line is evicted to update filter

    void notify_cache_evict(address_t evicted_addr) {
        #pragma HLS INLINE
        if constexpr (SPP_FILTER_ON) {
            spp_ghr_counter_t temp_issued = global_register.pf_issued;
            spp_ghr_counter_t temp_useful = global_register.pf_useful;
            
            prefetch_filter.check(evicted_addr, SPP_L2_EVICT,
                                temp_issued, temp_useful);
            
            global_register.pf_useful = temp_useful;
        }
    }

    // ========================================================================
    // Cache hit notification
    // ========================================================================
    // Called when demand request hits a prefetched line

    void notify_cache_hit(address_t hit_addr) {
        #pragma HLS INLINE
        if constexpr (SPP_FILTER_ON) {
            spp_ghr_counter_t temp_issued = global_register.pf_issued;
            spp_ghr_counter_t temp_useful = global_register.pf_useful;
            
            prefetch_filter.check(hit_addr, SPP_L2_DEMAND,
                                temp_issued, temp_useful);
            
            global_register.pf_useful = temp_useful;
        }
    }
};
