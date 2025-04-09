#pragma once
#include "global.hpp"


template<typename address_t, typename block_address_t, typename class_t>
struct ForwardingBufferEntry {
	bool valid;
	address_t inputBufferAddress;
	block_address_t lastAddress;
	class_t sequence[SEQUENCE_LENGTH];
	ForwardingBufferEntry(){}
};


template<typename address_t, typename block_address_t, typename class_t>
struct ForwardingBufferEntriesMatrix {
	ForwardingBufferEntry<address_t, block_address_t, class_t> entries[FORWARDING_DEPTH];
	ForwardingBufferEntriesMatrix(){}
};

template<typename address_t, typename block_address_t, typename class_t, typename burst_length_t>
struct BurstForwardingBufferEntry {
	bool valid;
	address_t inputBufferAddress;
	block_address_t lastAddress;
	class_t sequence[SEQUENCE_LENGTH];
	burst_length_t burstLengthSequence[SEQUENCE_LENGTH];
	ForwardingBufferEntry(){}
};


template<typename address_t, typename block_address_t, typename class_t, typename burst_length_t>
struct BurstForwardingBufferEntriesMatrix {
	BurstForwardingBufferEntry<address_t, block_address_t, class_t, burst_length_t> entries[FORWARDING_DEPTH];
	BurstForwardingBufferEntriesMatrix(){}
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


template<typename address_t, typename confidence_t, typename block_address_t, typename burst_length_t>
struct BurstConfidenceForwardingBufferEntry {
	bool valid;
	address_t inputBufferAddress;
	confidence_t confidence;
	confidence_t burstConfidence;
	block_address_t lastPredictedAddress;
	burst_length_t lastPredictedBurstLength;
	BurstConfidenceForwardingBufferEntry(){}
};


template<typename address_t, typename confidence_t, typename block_address_t, typename burst_length_t>
struct BurstConfidenceForwardingBufferEntriesMatrix {
	BurstConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t, burst_length_t> entries[FORWARDING_DEPTH];
	BUrstConfidenceForwardingBufferEntriesMatrix(){}
};


template<typename address_t, typename tag_t, typename block_address_t, typename class_t, typename lru_t>
class ForwardingBuffer {
public:
	ForwardingBufferEntry<address_t, block_address_t, class_t> read(
		ForwardingBufferEntry<address_t, block_address_t, class_t> entries[FORWARDING_DEPTH],
		address_t& address, forwarding_index_t& currentSlot, bool& isHit){
		#pragma HLS INLINE

		ForwardingBufferEntry<address_t, block_address_t, class_t> res;
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
		ForwardingBufferEntry<address_t, block_address_t, class_t> entries[FORWARDING_DEPTH],
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
		ForwardingBufferEntry<address_t, block_address_t, class_t> newEntry;
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

template<typename address_t, typename tag_t, typename block_address_t, typename class_t, typename burst_length_t, typename lru_t>
class BurstForwardingBuffer {
public:
	BurstForwardingBufferEntry<address_t, block_address_t, class_t, burst_length_t> read(
		BurstForwardingBufferEntry<address_t, block_address_t, class_t, burst_length_t> entries[FORWARDING_DEPTH],
		address_t& address, forwarding_index_t& currentSlot, bool& isHit){
		#pragma HLS INLINE

		BurstForwardingBufferEntry<address_t, block_address_t, class_t, burst_length_t> res;
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
		BurstForwardingBufferEntry<address_t, block_address_t, class_t, burst_length_t> entries[FORWARDING_DEPTH],
		block_address_t lastAddress, class_t sequence[SEQUENCE_LENGTH],
		burst_length_t burstSequence[SEQUENCE_LENGTH],
		address_t& address, forwarding_index_t& currentSlot, forwarding_index_t& nextSlot) {
		#pragma HLS INLINE

		forwarding_index_t nextSlot_ = nextSlot;
		if(nextSlot == (forwarding_index_t)(FORWARDING_DEPTH - 1)){
			nextSlot = 0;
		}
		else{
			nextSlot++;
		}
		BurstForwardingBufferEntry<address_t, block_address_t, class_t, burst_length_t> newEntry;
		newEntry.valid = true;
		newEntry.inputBufferAddress = address;
		newEntry.lastAddress = lastAddress;
		for(int k = 0; k < SEQUENCE_LENGTH; k++){
			#pragma HLS UNROLL
			newEntry.sequence[k] = sequence[k];
		}
		for(int k = 0; k < SEQUENCE_LENGTH; k++){
			#pragma HLS UNROLL
			newEntry.burstLengthSequence[k] = burstSequence[k];
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
		newEntry.inputBufferAddress = address;
		newEntry.confidence = confidence;
		newEntry.lastPredictedAddress = lastPredictedAddress;
		entries[nextSlot_] = newEntry;

		currentSlot = nextSlot_;
	}

};


template<typename address_t, typename confidence_t, typename block_address_t, typename burst_length_t>
class BurstConfidenceForwardingBuffer {
public:
	BurstConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t, burst_length_t> read(
			BurstConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t, burst_length_t> entries[FORWARDING_DEPTH],
		address_t& address, conf_forwarding_index_t& currentSlot, bool& isHit){
		#pragma HLS INLINE

		BurstConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t, burst_length_t> res;
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
		BurstConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t, burst_length_t> entries[FORWARDING_DEPTH],
		confidence_t confidence, block_address_t lastPredictedAddress,
		confidence_t burstConfidence, burst_length_t lastPredictedBurstLength,
		address_t& address, forwarding_index_t& currentSlot, forwarding_index_t& nextSlot) {
		#pragma HLS INLINE

		forwarding_index_t nextSlot_ = nextSlot;
		if(nextSlot == (forwarding_index_t)(FORWARDING_DEPTH - 1)){
			nextSlot = 0;
		}
		else{
			nextSlot++;
		}
		BurstConfidenceForwardingBufferEntry<address_t, confidence_t, block_address_t, burst_length_t> newEntry;
		newEntry.valid = true;
		newEntry.inputBufferAddress = address;
		newEntry.confidence = confidence;
		newEntry.lastPredictedAddress = lastPredictedAddress;
		newEntry.burstConfidence = burstConfidence;
		newEntry.lastPredictedBurstLength = lastPredictedBurstLength;
		entries[nextSlot_] = newEntry;

		currentSlot = nextSlot_;
	}

};



