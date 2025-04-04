#pragma once
#include "global.hpp"

template<typename confidence_t, typename block_address_t>
struct ConfidenceBufferEntry{
    confidence_t confidence;
	block_address_t lastPredictedAddress;
    ConfidenceBufferEntry(){}
};

template<typename confidence_t, typename block_address_t>
struct ConfidenceBufferEntriesMatrix {
		ConfidenceBufferEntry<confidence_t, block_address_t> entries[IB_NUM_SETS][IB_NUM_WAYS];
		ConfidenceBufferEntriesMatrix(){}
};

template<typename confidence_t, typename block_address_t>
class ConfidenceBuffer{
    public:
	ConfidenceBufferEntry<confidence_t, block_address_t> read(
		ConfidenceBufferEntry<confidence_t, block_address_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
		ib_index_t index, ib_way_t way);

	void write(
		ConfidenceBufferEntry<confidence_t, block_address_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
		ib_index_t index, ib_way_t way, ConfidenceBufferEntry<confidence_t, block_address_t>& entry);
};

template<typename confidence_t, typename block_address_t>
ConfidenceBufferEntry<confidence_t, block_address_t> 
ConfidenceBuffer<confidence_t, block_address_t>::read(
    ConfidenceBufferEntry<confidence_t, block_address_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
    ib_index_t index, ib_way_t way){
    #pragma HLS INLINE
#pragma HLS ARRAY_RESHAPE variable=entries dim=2 complete
#pragma HLS ARRAY_RESHAPE variable=entries dim=3 complete
// #pragma HLS ARRAY_PARTITION variable=entries dim=0 complete
#pragma HLS BIND_STORAGE variable=entries type=RAM_T2P impl=bram latency=1

	ConfidenceBufferEntry<confidence_t, block_address_t> res;
    res = entries[index][way];
    return res;
}

template<typename confidence_t, typename block_address_t>
void ConfidenceBuffer<confidence_t, block_address_t>::write(
    ConfidenceBufferEntry<confidence_t, block_address_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
    ib_index_t index, ib_way_t way, ConfidenceBufferEntry<confidence_t, block_address_t>& entry){
    #pragma HLS INLINE
#pragma HLS ARRAY_RESHAPE variable=entries dim=2 complete
#pragma HLS ARRAY_RESHAPE variable=entries dim=3 complete
// #pragma HLS ARRAY_PARTITION variable=entries dim=0 complete
#pragma HLS BIND_STORAGE variable=entries type=RAM_T2P impl=bram latency=1

    entries[index][way] = entry;
}
