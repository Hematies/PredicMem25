#pragma once

#include "spp_config.hpp"
#include "spp_data_type.hpp"

// ============================================================================
// SPP Prefetch Filter
// ============================================================================
// Tracks prefetched cache lines and their usefulness
// Prevents duplicate prefetch requests
// Updates global accuracy counters based on prefetch effectiveness

enum SPPFilterRequest {
    SPP_L2_PREFETCH = 0,      // L2 cache prefetch (high confidence)
    SPP_LLC_PREFETCH = 1,     // LLC prefetch (medium confidence)
    SPP_L2_DEMAND = 2,        // Demand request (check if prefetch was useful)
    SPP_L2_EVICT = 3          // Cache line eviction (cleanup)
};

// Forward declaration of matrix struct (defined in spp_init.hpp)
struct SPPPrefetchFilterMatrix;

template<typename filter_tag_t = spp_filter_tag_t>
class SPPPrefetchFilter {
public:
    filter_tag_t remainder_tag[SPP_FILTER_SET];
    spp_ghr_valid_t valid[SPP_FILTER_SET];      // Marked as prefetched
    spp_ghr_valid_t useful[SPP_FILTER_SET];     // Actually used

    // Default constructor - initialization via constexpr in caller
    SPPPrefetchFilter() = default;

    // Hash function
    static uint64_t hash_cache_line(uint64_t cache_line) {
        uint64_t key = cache_line;
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

    // Check filter and update counters
    // Returns: true if prefetch should proceed, false to skip
    spp_ghr_valid_t check(spp_address_t pf_addr, SPPFilterRequest filter_request,
                     spp_ghr_counter_t& pf_issued,
                     spp_ghr_counter_t& pf_useful) {
        spp_block_address_t cache_line = pf_addr >> SPP_LOG2_BLOCK_SIZE;
        uint64_t hash = hash_cache_line(cache_line);
        spp_filter_index_t quotient = (hash >> SPP_REMAINDER_BIT) & ((1 << SPP_QUOTIENT_BIT) - 1);
        filter_tag_t remainder = hash & ((1 << SPP_REMAINDER_BIT) - 1);

        spp_ghr_valid_t should_prefetch = 1;

        switch (filter_request) {
        case SPP_L2_PREFETCH:
            // Check if already prefetched
            if ((valid[quotient] || useful[quotient]) && 
                (remainder_tag[quotient] == remainder)) {
                // Duplicate prefetch request, skip
                should_prefetch = 0;
            } else {
                // New prefetch - mark as valid
                valid[quotient] = 1;
                useful[quotient] = 0;
                remainder_tag[quotient] = remainder;
            }
            break;

        case SPP_LLC_PREFETCH:
            // LLC prefetch with lower priority (don't set valid to allow L2 prefetch later)
            if ((valid[quotient] || useful[quotient]) && 
                (remainder_tag[quotient] == remainder)) {
                should_prefetch = 0;
            }
            // Note: don't set valid/useful for LLC prefetch
            break;

        case SPP_L2_DEMAND:
            // Demand hit - mark if prefetch was useful
            if ((remainder_tag[quotient] == remainder) && !useful[quotient]) {
                useful[quotient] = 1;
                
                // Update counters if this was a prefetched line
                if (valid[quotient]) {
                    // Saturating increment
                    if (pf_useful < SPP_GLOBAL_COUNTER_MAX) {
                        pf_useful++;
                    }
                }
            }
            should_prefetch = 0;
            break;

        case SPP_L2_EVICT:
            // Line eviction - decrease counters if prefetched but not used
            if (valid[quotient] && !useful[quotient] && (pf_useful > 0)) {
                pf_useful--;
            }
            
            // Reset filter entry
            valid[quotient] = 0;
            useful[quotient] = 0;
            remainder_tag[quotient] = 0;
            should_prefetch = 0;
            break;

        default:
            should_prefetch = 0;
            break;
        }

        return should_prefetch;
    }
};
