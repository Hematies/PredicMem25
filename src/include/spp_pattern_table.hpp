#pragma once

#include "spp_config.hpp"
#include "spp_signature_table.hpp"
#include "spp_data_type.hpp"

// ============================================================================
// SPP Pattern Table (PT)
// ============================================================================
// Stores correlation between signatures and deltas
// Each entry contains:
// - delta: signed offset delta
// - c_delta: confidence counter for this delta
// - c_sig: total confidence for all deltas at this signature

// Forward declaration of matrix struct (defined in spp_init.hpp)
// struct SPPPatternTableMatrix;

template<typename pt_delta_t = spp_pt_delta_t, typename pt_confidence_t = spp_pt_confidence_t>
class SPPPatternTable {
public:
    pt_delta_t delta[SPP_PT_SET][SPP_PT_WAY];
    pt_confidence_t c_delta[SPP_PT_SET][SPP_PT_WAY];  // Confidence counter per delta
    pt_confidence_t c_sig[SPP_PT_SET];                 // Total confidence per signature

    // Default constructor - initialization via constexpr in caller
    SPPPatternTable() = default;

    // Hash function (same as used in ST)
    static uint64_t hash_signature(uint64_t sig) {
        uint64_t key = sig;
        key += (key << 12);
        key ^= (key >> 22);
        key += (key << 4);
        key ^= (key >> 9);
        key += (key << 10);
        key ^= (key >> 2);
        key += (key << 7);
        key ^= (key >> 12);
        key = (key >> 3) * 2654435761ULL;
        return key;
    }

    // Update pattern table with (signature, delta) pair
    // Called when we observe a delta from a previous signature
    void update_pattern(uint32_t last_sig, pt_delta_t curr_delta) {
        spp_pt_set_index_t set = hash_signature(last_sig) % SPP_PT_SET;
        spp_pt_way_index_t match = SPP_PT_WAY;
        spp_pt_way_index_t victim_way = SPP_PT_WAY;
        pt_confidence_t min_counter = SPP_C_DELTA_MAX + 1;

        // Search for matching delta entry
        #pragma HLS UNROLL
        for (uint32_t way = 0; way < SPP_PT_WAY; way++) {
            if (delta[set][way] == curr_delta) {
                match = way;
                break;
            }
            // Track minimum confidence for replacement
            if (c_delta[set][way] < min_counter) {
                min_counter = c_delta[set][way];
                victim_way = way;
            }
        }

        // Update or allocate entry
        if (match < SPP_PT_WAY) {
            // Hit: increment confidence of existing entry
            if (c_delta[set][match] < SPP_C_DELTA_MAX) {
                c_delta[set][match]++;
            }
        } else if (victim_way < SPP_PT_WAY) {
            // Miss: replace victim entry
            delta[set][victim_way] = curr_delta;
            c_delta[set][victim_way] = 0;
        }

        // Update global confidence counter with saturation
        if (c_sig[set] < SPP_C_SIG_MAX) {
            c_sig[set]++;
        } else {
            // Half all confidence values when saturated
            #pragma HLS UNROLL
            for (uint32_t way = 0; way < SPP_PT_WAY; way++) {
                c_delta[set][way] >>= 1;
            }
            c_sig[set] >>= 1;
        }
    }

    // Read pattern table and generate prefetch candidates
    // Returns candidate deltas with their confidence scores
    // Also performs lookahead to enable speculative prefetching
    void read_pattern(uint32_t curr_sig, 
                     pt_delta_t* delta_q, 
                     pt_confidence_t* confidence_q,
                     spp_pt_way_index_t& lookahead_way,
                     pt_confidence_t& lookahead_conf,
                     uint32_t& pf_q_tail,
                     uint32_t& depth,
                     spp_accuracy_t global_accuracy) {
        spp_pt_set_index_t set = hash_signature(curr_sig) % SPP_PT_SET;
        pt_confidence_t local_conf = 0;
        pt_confidence_t pf_conf = 0;
        pt_confidence_t max_conf = 0;
        
        lookahead_way = SPP_PT_WAY;
        lookahead_conf = 0;

        if (c_sig[set] > 0) {
            #pragma HLS UNROLL
            for (uint32_t way = 0; way < SPP_PT_WAY; way++) {
                // Calculate local confidence: ratio of this delta to total
                local_conf = (100 * c_delta[set][way]) / c_sig[set];
                
                // Calculate prefetch confidence with lookahead damping
                if (depth > 0) {
                    // Speculative prefetch uses global accuracy to dampen confidence
                    pf_conf = (global_accuracy * c_delta[set][way]) / c_sig[set];
                    pf_conf = (pf_conf * lookahead_conf) / 100;
                } else {
                    pf_conf = local_conf;
                }

                // Generate prefetch if confidence exceeds threshold
                if (pf_conf >= SPP_PF_THRESHOLD) {
                    confidence_q[pf_q_tail] = pf_conf;
                    delta_q[pf_q_tail] = delta[set][way];
                    pf_q_tail++;

                    // Track highest confidence for lookahead
                    if (pf_conf > max_conf) {
                        lookahead_way = way;
                        max_conf = pf_conf;
                    }
                }
            }
            // Add separator in queue
            confidence_q[pf_q_tail] = 0;
            pf_q_tail++;

            // Update lookahead info for next iteration
            lookahead_conf = max_conf;
            if (lookahead_conf >= SPP_PF_THRESHOLD) {
                depth++;
            }
        }
    }
};
