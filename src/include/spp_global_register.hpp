#pragma once

#include "spp_config.hpp"
#include "spp_data_type.hpp"

// ============================================================================
// Global History Register (GHR) for SPP
// ============================================================================
// Stores information about prefetch requests that cross page boundaries
// to bootstrap SPP learning when accessing new pages

// Forward declaration of storage struct (defined in spp_init.hpp)
struct SPPGlobalRegisterStorage;

template<typename st_tag_t = spp_st_tag_t, typename st_sig_t = spp_st_sig_t, typename st_confidence_t = spp_st_confidence_t, typename st_delta_t = spp_pt_delta_t>
class SPPGlobalRegister {
public:
    // Global accuracy tracking
    spp_ghr_counter_t pf_issued;
    spp_ghr_counter_t pf_useful;
    spp_accuracy_t global_accuracy;

    // GHR entries
    spp_ghr_valid_t valid[SPP_MAX_GHR_ENTRY];
    st_sig_t sig[SPP_MAX_GHR_ENTRY];
    st_confidence_t confidence[SPP_MAX_GHR_ENTRY];
    spp_ghr_offset_t offset[SPP_MAX_GHR_ENTRY];  // Page offset for matching
    st_delta_t delta[SPP_MAX_GHR_ENTRY];

    // Default constructor - initialization via constexpr in caller
    SPPGlobalRegister() = default;

    // Update global accuracy counter
    void update_global_accuracy() {
        if (pf_issued > 0) {
            global_accuracy = (100 * pf_useful) / pf_issued;
        } else {
            global_accuracy = 0;
        }
    }

    // Saturating counter increment
    void increment_pf_issued() {
        if (pf_issued < SPP_GLOBAL_COUNTER_MAX) {
            pf_issued++;
        }
    }

    // Saturating counter decrement for pf_useful
    void decrement_pf_useful() {
        if (pf_useful > 0) {
            pf_useful--;
        }
    }

    // Update or create a GHR entry for cross-page prefetch
    void update_entry(st_sig_t pf_sig, st_confidence_t pf_confidence, 
                      spp_ghr_offset_t pf_offset, st_delta_t pf_delta) {
        st_confidence_t min_conf = 100;
        spp_ghr_way_index_t victim_way = SPP_MAX_GHR_ENTRY;

        // Search for matching offset or minimum confidence entry
        #pragma HLS UNROLL
        for (spp_ghr_way_index_t i = 0; i < SPP_MAX_GHR_ENTRY; i++) {
            // Check if offset matches - update existing entry
            if (valid[i] && (offset[i] == pf_offset)) {
                sig[i] = pf_sig;
                confidence[i] = pf_confidence;
                delta[i] = pf_delta;
                return;
            }

            // Track minimum confidence for replacement policy
            if (confidence[i] < min_conf) {
                min_conf = confidence[i];
                victim_way = i;
            }
        }

        // If no matching entry found, replace victim with lowest confidence
        if (victim_way < SPP_MAX_GHR_ENTRY) {
            valid[victim_way] = 1;
            sig[victim_way] = pf_sig;
            confidence[victim_way] = pf_confidence;
            offset[victim_way] = pf_offset;
            delta[victim_way] = pf_delta;
        }
    }

    // Check if there's a matching GHR entry for given page offset
    // Returns way index if found, SPP_MAX_GHR_ENTRY if not found
    spp_ghr_way_index_t check_entry(spp_ghr_offset_t page_offset) {
        st_confidence_t max_conf = 0;
        spp_ghr_way_index_t max_conf_way = SPP_MAX_GHR_ENTRY;

        #pragma HLS UNROLL
        for (spp_ghr_way_index_t i = 0; i < SPP_MAX_GHR_ENTRY; i++) {
            if (valid[i] && (offset[i] == page_offset) && 
                (confidence[i] > max_conf)) {
                max_conf = confidence[i];
                max_conf_way = i;
            }
        }

        return max_conf_way;
    }
};
