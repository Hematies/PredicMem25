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

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
class InputBuffer {
protected:
	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> (* entries)[IB_NUM_WAYS];

	void updateLRU(index_t index, way_t way);
	way_t getLeastRecentWay(index_t index);
	way_t queryWay(index_t index, tag_t tag);

public:
	InputBuffer(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS]) : entries(entries){}
	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> operator()(bool opRead, address_t address, InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entry);
};

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
void InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>::updateLRU(index_t index, way_t way)
{
	bool areAllWaysSaturated = true;
	for (int w = 0; w < IB_NUM_WAYS; w++) {
		areAllWaysSaturated = areAllWaysSaturated && (this->entries[index][way].lruCounter == IB_MAX_LRU_COUNTER);
	}

	for (int w = 0; w < IB_NUM_WAYS; w++) {
		bool increment = w == way;
		bool reset = areAllWaysSaturated;
		bool isSaturated = this->entries[index][way].lruCounter == IB_MAX_LRU_COUNTER;
		if (reset) {
			if (!increment) this->entries[index][way].lruCounter = 0;
			else this->entries[index][way].lruCounter = 1;
		}
		else {
			if (increment && !isSaturated) this->entries[index][way].lruCounter++;
		}

	}
}

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
way_t InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>::getLeastRecentWay(index_t index)
{
	way_t res = 0;
	lru_t leastRecency = IB_MAX_LRU_COUNTER;
	for (int w = 0; w < IB_NUM_WAYS; w++) {
		if (entries[index][w].lruCounter < leastRecency) {
			res = w;
			leastRecency = entries[index][w].lruCounter;
		}
	}
	return res;
}

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
way_t InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>::queryWay(index_t index, tag_t tag)
{
	way_t res = IB_NUM_WAYS;
	for (int w = 0; w < IB_NUM_WAYS; w++) {
		if (entries[index][w].tag == tag && entries[index][w].valid)
			res = w;
	}
	return res;
}

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>::
operator()(bool opRead, address_t address, InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entry) {
	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> res = 
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>();

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	tag_t tag = address >> numIndexBits;
	index_t index = address % (1 << numIndexBits);
	way_t way = this->queryWay(index, tag);

	if (opRead) {
		
		if (way != IB_NUM_WAYS) {
			res = this->entries[index][way];
		}
	}
	else {
		if (way == IB_NUM_WAYS) {
			way = this->getLeastRecentWay(index);
		}
		else {
			entry.lruCounter = this->entries[index][way].lruCounter;
		}
		res = entry;
		this->entries[index][way] = entry;
	}

	this->updateLRU(index, way);
	return res;
}
