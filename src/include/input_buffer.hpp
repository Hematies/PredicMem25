#pragma once
#include "global.hpp"

template<typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
struct InputBufferEntry {
	bool valid;
	tag_t tag;
	block_address_t lastAddress;
	class_t sequence[SEQUENCE_LENGTH];
	confidence_t confidence;
	block_address_t lastPredictedAddress;
	lru_t lruCounter;
};


template<typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
struct InputBufferEntriesMatrix {
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS];
};


template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
class InputBuffer {
protected:

	void updateLRU(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS], index_t index, way_t way);
	way_t getLeastRecentWay(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS], index_t index);
	way_t queryWay(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS], index_t index, tag_t tag);

public:
	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> read(
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
		address_t address, bool& isHit);
	// InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>
	void write(
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
		address_t address, InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entry);

};

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
void InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>
	::updateLRU(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS], index_t index, way_t way)
{
#pragma HLS INLINE
	bool areAllWaysSaturated = true;
	for (int w = 0; w < IB_NUM_WAYS; w++) {
#pragma HLS UNROLL
		areAllWaysSaturated = areAllWaysSaturated && (entries[index][way].lruCounter == IB_MAX_LRU_COUNTER);
	}

	lru_t lruCounters[IB_NUM_WAYS];

	for (int w = 0; w < IB_NUM_WAYS; w++) {
#pragma HLS UNROLL
		bool increment = w == way;
		bool reset = areAllWaysSaturated;
		lru_t lruCounter = entries[index][way].lruCounter;
		bool isSaturated = lruCounter == IB_MAX_LRU_COUNTER;

		if (reset) {
			if (!increment) lruCounter = 0;
			else lruCounter = 1;
		}
		else {
			if (increment && !isSaturated) lruCounter++;
		}
		lruCounters[w] = lruCounter;

	}

	for (int w = 0; w < IB_NUM_WAYS; w++) {
#pragma HLS UNROLL
		entries[index][way].lruCounter = lruCounters[w];
	}
}

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
way_t InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>
::getLeastRecentWay(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS], index_t index)
{
#pragma HLS INLINE
	way_t res = 0;
	lru_t leastRecency = IB_MAX_LRU_COUNTER;
	for (int w = 0; w < IB_NUM_WAYS; w++) {
#pragma HLS UNROLL
		if (entries[index][w].lruCounter < leastRecency) {
			res = w;
			leastRecency = entries[index][w].lruCounter;
		}
	}
	return res;
}

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
way_t InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>::queryWay(
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
		index_t index, tag_t tag)
{
	#pragma HLS INLINE
	way_t res = IB_NUM_WAYS;

	for (int w = 0; w < IB_NUM_WAYS; w++) {
#pragma HLS UNROLL
		if (entries[index][w].tag == tag && entries[index][tag].valid)
			res = w;
	}
	return res;
}

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>::
read(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS], address_t inputBufferAddress, bool& isHit) {
// #pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=entries dim=0 factor=2 block

	#pragma HLS PIPELINE
	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> res = 
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>();

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	tag_t tag = inputBufferAddress >> numIndexBits;
	index_t index = inputBufferAddress % (1 << numIndexBits);
	isHit = false;
	way_t way = this->queryWay(entries, index, tag);

	if (way != IB_NUM_WAYS) {
		res = entries[index][way];
		isHit = true;
	}
	this->updateLRU(entries, index, way);


	return res;
}


template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
// InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>
void
InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>::
write(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
	address_t inputBufferAddress, InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entry) {
// #pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=entries dim=0 factor=2 block
#pragma HLS PIPELINE
	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> res =
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>();

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	tag_t tag = inputBufferAddress >> numIndexBits;
	index_t index = inputBufferAddress % (1 << numIndexBits);
	way_t way = this->queryWay(entries, index, tag);


	if (way == IB_NUM_WAYS) {
		way = this->getLeastRecentWay(entries, index);
		entries[index][way].tag = entry.tag;
		entries[index][way].valid = entry.valid;
	}

	entries[index][way].lastAddress = entry.lastAddress;
	entries[index][way].confidence = entry.confidence;
	entries[index][way].lastPredictedAddress = entry.lastPredictedAddress;
	for(int i = 0; i < SEQUENCE_LENGTH; i++){
#pragma HLS UNROLL
		entries[index][way].sequence[i] = entry.sequence[i];
	}

	this->updateLRU(entries, index, way);

}

