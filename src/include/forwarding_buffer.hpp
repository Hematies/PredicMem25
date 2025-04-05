#pragma once
#include "global.hpp"


template<typename address_t, typename block_address_t, typename class_t, typename confidence_t>
struct ForwardingBufferEntry {
	bool valid;
	address_t inputBufferAddress;
	block_address_t lastAddress;
	class_t sequence[SEQUENCE_LENGTH];
	ForwardingBufferEntry(){}
};


template<typename address_t, typename block_address_t, typename class_t, typename confidence_t>
struct ForwardingBufferEntriesMatrix {
	ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> entries[FORWARDING_DEPTH];
	ForwardingBufferEntriesMatrix(){}
};

template<typename address_t, typename confidence_t, typename block_address_t>
struct ConfidenceForwardingBufferEntry {
	bool valid;
	address_t inputBufferAddress;
	confidence_t confidence;
	block_address_t lastPredictedAddress;
	ConfidenceForwardingBufferEntry(){}
};


template<typename address_t, typename confidence_t, typename block_address_t>
struct ConfidenceForwardingBufferEntriesMatrix {
	ConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t> entries[FORWARDING_DEPTH];
	ConfidenceForwardingBufferEntriesMatrix(){}
};


template<typename address_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
class ForwardingBuffer {
public:
	ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> read(
		ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> entries[FORWARDING_DEPTH],
		address_t& address, forwarding_index_t& currentSlot, bool& isHit){
		#pragma HLS INLINE

		ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> res;
		isHit = false;

		for(int i = 0; i < FORWARDING_DEPTH; i++){
			#pragma HLS UNROLL
			if(entries[i].valid && entries[i].inputBufferAddress == address){
				isHit = true;
				res = entries[i];
				currentSlot = i;
				break;
			}
		}

		return res;
	}

	void write(
		ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> entries[FORWARDING_DEPTH],
		block_address_t lastAddress, class_t sequence[SEQUENCE_LENGTH],
		address_t& address, forwarding_index_t& currentSlot, forwarding_index_t& nextSlot) {
		#pragma HLS INLINE

		forwarding_index_t nextSlot_ = nextSlot;
		if(nextSlot == (forwarding_index_t)(FORWARDING_DEPTH - 1)){
			nextSlot = 0;
		}
		else{
			nextSlot++;
		}
		ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> newEntry;
		newEntry.valid = true;
		newEntry.inputBufferAddress = address;
		newEntry.lastAddress = lastAddress;
		for(int k = 0; k < SEQUENCE_LENGTH; k++){
			#pragma HLS UNROLL
			newEntry.sequence[k] = sequence[k];
		}
		entries[nextSlot_] = newEntry;

		currentSlot = nextSlot_;
	}

};

template<typename address_t, typename confidence_t, typename block_address_t>
class ConfidenceForwardingBuffer {
public:
	ConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t> read(
			ConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t> entries[FORWARDING_DEPTH],
		address_t& address, conf_forwarding_index_t& currentSlot, bool& isHit){
		#pragma HLS INLINE

		ConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t> res;
		isHit = false;

		for(int i = 0; i < FORWARDING_DEPTH; i++){
			#pragma HLS UNROLL
			if(entries[i].valid && entries[i].inputBufferAddress == address){
				isHit = true;
				res = entries[i];
				currentSlot = i;
				break;
			}
		}

		return res;
	}

	void write(
		ConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t> entries[FORWARDING_DEPTH],
		confidence_t confidence, block_address_t lastPredictedAddress,
		address_t& address, forwarding_index_t& currentSlot, forwarding_index_t& nextSlot) {
		#pragma HLS INLINE

		forwarding_index_t nextSlot_ = nextSlot;
		if(nextSlot == (forwarding_index_t)(FORWARDING_DEPTH - 1)){
			nextSlot = 0;
		}
		else{
			nextSlot++;
		}
		ConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t> newEntry;
		newEntry.valid = true;
		newEntry.confidence = confidence;
		newEntry.lastPredictedAddress = lastPredictedAddress;
		entries[nextSlot_] = newEntry;

		currentSlot = nextSlot_;
	}

};




