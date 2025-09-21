#include "global.hpp"

template<typename block_address_t, typename pb_index_t, typename pb_tag_t>
class PrefetchBuffer {
public:
    void operator()(block_address_t prefetchAddress, bool &issuePrefetch) {
        static pb_tag_t table[PB_NUM_SETS] = {0};

        #pragma HLS ARRAY_PARTITION variable=table complete dim=0
        #pragma HLS DEPENDENCE array false variable=table
        
        pb_index_t index = prefetchAddress % (1 << PB_NUM_SETS_LOG2);
        pb_tag_t tag = prefetchAddress >> PB_NUM_SETS_LOG2;

        if (table[index] == tag) {
            issuePrefetch = false; // Already in the buffer, do not prefetch
        } else {
            issuePrefetch = true; // Not in the buffer, prefetch
            table[index] = tag; // Update the buffer with the new tag
        }

    }
    
};