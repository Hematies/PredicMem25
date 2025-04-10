#pragma once
#include "global.hpp"

template<typename confidence_t, typename block_address_t, typename burst_length_t>
struct BurstConfidenceBufferEntry{
    confidence_t confidence;
    confidence_t burstConfidence;
	block_address_t lastPredictedAddress;
    burst_length_t lastPredictedBurstLength;
    BurstConfidenceBufferEntry(){}
};

template<typename confidence_t, typename block_address_t, typename burst_length_t>
struct BurstConfidenceBufferEntriesMatrix {
		BurstConfidenceBufferEntry<confidence_t, block_address_t, burst_length_t> entries[IB_NUM_SETS][IB_NUM_WAYS];
		ConfidenceBufferEntriesMatrix(){}
};

template<typename confidence_t, typename block_address_t, typename burst_length_t>
class BurstConfidenceBuffer{
    public:
	BurstConfidenceBufferEntry<confidence_t, block_address_t, burst_length_t> read(
		BurstConfidenceBufferEntry<confidence_t, block_address_t, burst_length_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
		ib_index_t index, ib_way_t way);

	void write(
		BurstConfidenceBufferEntry<confidence_t, block_address_t, burst_length_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
		ib_index_t index, ib_way_t way, BurstConfidenceBufferEntry<confidence_t, block_address_t, burst_length_t>& entry);
};

template<typename confidence_t, typename block_address_t, typename burst_length_t>
BurstConfidenceBufferEntry<confidence_t, block_address_t, burst_length_t> 
BurstConfidenceBuffer<confidence_t, block_address_t, burst_length_t>::read(
    BurstConfidenceBufferEntry<confidence_t, block_address_t, burst_length_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
    ib_index_t index, ib_way_t way){
    #pragma HLS INLINE
#pragma HLS ARRAY_RESHAPE variable=entries dim=2 complete
#pragma HLS ARRAY_RESHAPE variable=entries dim=3 complete
// #pragma HLS ARRAY_PARTITION variable=entries dim=0 complete
#pragma HLS BIND_STORAGE variable=entries type=RAM_T2P impl=bram latency=1

	BurstConfidenceBufferEntry<confidence_t, block_address_t, burst_length_t> res;
    res = entries[index][way];
    return res;
}

template<typename confidence_t, typename block_address_t, typename burst_length_t>
void BurstConfidenceBuffer<confidence_t, block_address_t, burst_length_t>::write(
    BurstConfidenceBufferEntry<confidence_t, block_address_t, burst_length_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
    ib_index_t index, ib_way_t way, BurstConfidenceBufferEntry<confidence_t, block_address_t, burst_length_t>& entry){
    #pragma HLS INLINE
#pragma HLS ARRAY_RESHAPE variable=entries dim=2 complete
#pragma HLS ARRAY_RESHAPE variable=entries dim=3 complete
// #pragma HLS ARRAY_PARTITION variable=entries dim=0 complete
#pragma HLS BIND_STORAGE variable=entries type=RAM_T2P impl=bram latency=1

    entries[index][way] = entry;
}
