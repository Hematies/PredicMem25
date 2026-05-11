#pragma once

#include "spp_config.hpp"
#include "spp_data_type.hpp"

// ============================================================================
// SPP Signature Table (ST)
// ============================================================================
// Stores per-page information including:
// - Page tag (partial page address)
// - Signature for pattern correlation
// - Last cache block offset within the page
// - LRU replacement information

// Forward declaration of matrix struct (defined in spp_init.hpp)
struct SPPSignatureTableMatrix;

template<typename st_tag_t = spp_st_tag_t, typename st_sig_t = spp_st_sig_t, typename st_confidence_t = spp_st_confidence_t>
class SPPSignatureTable {
public:
    spp_ghr_valid_t valid[SPP_ST_SET][SPP_ST_WAY];
    st_tag_t tag[SPP_ST_SET][SPP_ST_WAY];
    spp_page_offset_t last_offset[SPP_ST_SET][SPP_ST_WAY];  // Last cache line offset in page
    st_sig_t sig[SPP_ST_SET][SPP_ST_WAY];            // Current signature
    spp_st_lru_t lru[SPP_ST_SET][SPP_ST_WAY];        // LRU counter

    // Default constructor - initialization via constexpr in caller
    SPPSignatureTable() = default;

    // Hash function for address-to-set mapping (Robert Jenkins' 32-bit mix)
    static uint64_t hash_address(uint64_t key) {
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

    // Read and update signature based on new page access
    // Returns: last_sig (previous signature), curr_sig (new signature), delta (offset difference)
    void read_and_update_sig(spp_address_t page, spp_page_offset_t page_offset,
                            st_sig_t& last_sig, st_sig_t& curr_sig, 
                            spp_pt_delta_t& delta) {
        spp_st_set_index_t set = hash_address(page) % SPP_ST_SET;
        st_tag_t partial_page = page & SPP_ST_TAG_MASK;
        spp_st_way_index_t match = SPP_ST_WAY;
        spp_ghr_valid_t st_hit = 0;

        // Stage 1: Search for matching tag
        #pragma HLS UNROLL
        for (uint32_t way = 0; way < SPP_ST_WAY; way++) {
            if (valid[set][way] && (tag[set][way] == partial_page)) {
                match = way;
                break;
            }
        }

        // Stage 2: If hit, calculate delta and update signature
        if (match < SPP_ST_WAY) {
            last_sig = sig[set][match];
            delta = page_offset - last_offset[set][match];

            if (delta != 0) {
                // Generate signature delta with 7-bit sign magnitude representation
                spp_sig_delta_t sig_delta = (delta < 0) ? 
                    (((-delta) & 0x3F) | 0x40) : delta;
                
                sig[set][match] = ((last_sig << SPP_SIG_SHIFT) ^ sig_delta) & SPP_SIG_MASK;
            }
            
            curr_sig = sig[set][match];
            last_offset[set][match] = page_offset;
            st_hit = 1;
        } else {
            // Stage 3: If miss, find replacement way (invalid or LRU victim)
            match = SPP_ST_WAY;
            
            // First, try to find invalid entry
            #pragma HLS UNROLL
            for (uint32_t way = 0; way < SPP_ST_WAY; way++) {
                if (!valid[set][way]) {
                    match = way;
                    break;
                }
            }

            // If no invalid entry, find LRU victim
            if (match == SPP_ST_WAY) {
                #pragma HLS UNROLL
                for (uint32_t way = 0; way < SPP_ST_WAY; way++) {
                    if (lru[set][way] == (SPP_ST_WAY - 1)) {
                        match = way;
                        break;
                    }
                }
            }

            // Allocate new entry
            if (match < SPP_ST_WAY) {
                valid[set][match] = 1;
                tag[set][match] = partial_page;
                sig[set][match] = 0;
                last_offset[set][match] = page_offset;
                curr_sig = 0;
                last_sig = 0;
                delta = 0;
            }
        }

        // Stage 4: Update LRU
        if (match < SPP_ST_WAY) {
            #pragma HLS UNROLL
            for (uint32_t way = 0; way < SPP_ST_WAY; way++) {
                if (lru[set][way] < lru[set][match]) {
                    lru[set][way]++;
                }
            }
            lru[set][match] = 0;  // Promote to MRU
        }
    }
};
