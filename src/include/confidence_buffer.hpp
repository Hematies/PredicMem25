#pragma once
#include "global.hpp"

template<typename confidence_t, typename block_address_t>
struct ConfidenceBufferEntry{
    confidence_t confidence;
	block_address_t lastPredictedAddress;
    ConfidenceBufferEntry(){}
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
    #pragma HLS ARRAY_RESHAPE variable=entries dim=0 complete

    return entries[index][way];
}

template<typename confidence_t, typename block_address_t>
void ConfidenceBuffer<confidence_t, block_address_t>::write(
    ConfidenceBufferEntry<confidence_t, block_address_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
    ib_index_t index, ib_way_t way, ConfidenceBufferEntry<confidence_t, block_address_t>& entry){
    #pragma HLS INLINE
    #pragma HLS ARRAY_RESHAPE variable=entries dim=0 complete

    entries[index][way] = entry;
}
