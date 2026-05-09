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
    // Processes a cache access and generates prefetch requests
    //
    // Input:
    //   - addr: Physical memory address
    //   - cache_hit: Whether this was a cache hit or miss
    //   - useful_prefetch: Whether previous prefetch was useful (feedback)
    //
    // Output:
    //   - prefetch_deltas: Array of cache line offsets to prefetch
    //   - prefetch_confidences: Confidence scores for each prefetch
    //   - num_prefetches: Number of valid prefetch requests generated
    //   - num_prefetches_l2: Number of L2-level prefetches (high confidence)

    void process_cache_access(address_t addr,
                             ap_uint<1> cache_hit,
                             ap_uint<1> useful_prefetch,
                             pt_delta_t* prefetch_deltas,
                             pt_confidence_t* prefetch_confidences,
                             uint32_t& num_prefetches,
                             uint32_t& num_prefetches_l2) {
        #pragma HLS PIPELINE
        
        // Update global accuracy
        global_register.update_global_accuracy();

        // Extract page and offset information
        uint32_t page = addr >> SPP_LOG2_PAGE_SIZE;
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

        // Also check filter for this demand request
        global_register.global_accuracy = 
            (global_register.pf_issued > 0) ? 
            ((100 * global_register.pf_useful) / global_register.pf_issued) : 0;

        // ====================================================================
        // Stage 2: Pattern Table Update
        // ====================================================================
        // Update pattern correlation if we have a previous signature
        if (last_sig != 0) {
            pattern_table.update_pattern(last_sig, delta);
        }

        // ====================================================================
        // Stage 3: Prefetch Generation
        // ====================================================================
        // Generate prefetch candidates using lookahead
        block_address_t base_addr = addr & ~(SPP_BLOCK_SIZE - 1);
        uint32_t pf_queue_head = 0;
        uint32_t pf_queue_tail = 0;
        pt_delta_t delta_queue[SPP_MAX_PREFETCH_QUEUE];
        pt_confidence_t confidence_queue[SPP_MAX_PREFETCH_QUEUE];

        num_prefetches = 0;
        num_prefetches_l2 = 0;

        // Initialize first prefetch queue entry
        confidence_queue[0] = 100;
        pf_queue_tail = 1;

        uint32_t lookahead_depth = 0;
        ap_uint<1> do_lookahead = 1;

        // Lookahead loop - generate speculative prefetch sequences
        while (do_lookahead && lookahead_depth < 3) {  // Max 3 lookaheads
            uint32_t lookahead_way = SPP_PT_WAY;
            pt_confidence_t lookahead_conf = 0;
            uint32_t pf_q_start = pf_queue_head;

            // Read patterns for current signature
            pattern_table.read_pattern(curr_sig, 
                                       delta_queue, 
                                       confidence_queue,
                                       lookahead_way,
                                       lookahead_conf,
                                       pf_queue_tail,
                                       lookahead_depth,
                                       global_register.global_accuracy);

            do_lookahead = 0;

            // Process all prefetches in current queue batch
            for (uint32_t i = pf_q_start; i < pf_queue_tail && 
                 num_prefetches < SPP_MAX_PREFETCH_QUEUE; i++) {
                if (confidence_queue[i] >= SPP_PF_THRESHOLD) {
                    // Calculate prefetch address
                    block_address_t pf_addr = base_addr + 
                                              (delta_queue[i] << SPP_LOG2_BLOCK_SIZE);

                    // Check if within same page (page boundary protection)
                    if ((addr & ~(SPP_PAGE_SIZE - 1)) == 
                        (pf_addr & ~(SPP_PAGE_SIZE - 1))) {
                        
                        // Determine L2 vs LLC prefetch based on confidence
                        SPPFilterRequest request_type = 
                            (confidence_queue[i] >= SPP_FILL_THRESHOLD) ? 
                            SPP_L2_PREFETCH : SPP_LLC_PREFETCH;

                        // Check filter and issue if allowed
                        ap_uint<1> should_prefetch = 
                            prefetch_filter.check(pf_addr, request_type,
                                                global_register.pf_issued,
                                                global_register.pf_useful);

                        if (should_prefetch) {
                            prefetch_deltas[num_prefetches] = delta_queue[i];
                            prefetch_confidences[num_prefetches] = confidence_queue[i];
                            
                            if (request_type == SPP_L2_PREFETCH) {
                                num_prefetches_l2++;
                                // Increment prefetch issued counter
                                global_register.increment_pf_issued();
                                if (global_register.pf_issued > SPP_GLOBAL_COUNTER_MAX) {
                                    global_register.pf_issued >>= 1;
                                    global_register.pf_useful >>= 1;
                                }
                            }
                            
                            num_prefetches++;
                        }
                    } else {
                        // Cross-page prefetch - store in GHR for future learning
                        if constexpr (SPP_GHR_ON) {
                            ap_uint<6> pf_offset = (pf_addr >> SPP_LOG2_BLOCK_SIZE) & 0x3F;
                            global_register.update_entry(curr_sig, 
                                                        confidence_queue[i],
                                                        pf_offset,
                                                        delta_queue[i]);
                        }
                    }

                    do_lookahead = 1;
                }
            }

            // Update base address and signature for next lookahead iteration
            if (lookahead_way < SPP_PT_WAY) {
                uint32_t pt_set = SPPPatternTable<pt_delta_t, 
                                                  pt_confidence_t>::hash_signature(curr_sig) % 
                                  SPP_PT_SET;
                pt_delta_t lookahead_delta = pattern_table.delta[pt_set][lookahead_way];
                
                base_addr += (lookahead_delta << SPP_LOG2_BLOCK_SIZE);

                // Update signature for next iteration
                spp_sig_delta_t sig_delta = (lookahead_delta < 0) ? 
                    (((-lookahead_delta) & 0x3F) | 0x40) : lookahead_delta;
                curr_sig = ((curr_sig << SPP_SIG_SHIFT) ^ sig_delta) & SPP_SIG_MASK;
                
                lookahead_depth++;
            } else {
                do_lookahead = 0;
            }

            if (!SPP_LOOKAHEAD_ON) {
                do_lookahead = 0;
            }
        }
    }

    // ========================================================================
    // Cache fill notification
    // ========================================================================
    // Called when a cache line is evicted to update filter

    void notify_cache_evict(address_t evicted_addr) {
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
        if constexpr (SPP_FILTER_ON) {
            spp_ghr_counter_t temp_issued = global_register.pf_issued;
            spp_ghr_counter_t temp_useful = global_register.pf_useful;
            
            prefetch_filter.check(hit_addr, SPP_L2_DEMAND,
                                temp_issued, temp_useful);
            
            global_register.pf_useful = temp_useful;
        }
    }
};
