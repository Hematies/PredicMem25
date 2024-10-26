#pragma once
#include "global.hpp"

template<typename delta_t, typename confidence_t>
struct DictionaryEntry {
	bool valid;
	delta_t delta;
	confidence_t confidence;
};

template<typename index_t, typename delta_t, typename confidence_t> 
class Dictionary {
protected:
	void updateConfidence(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], index_t index);
	index_t getIndexOfDelta(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], delta_t delta);
	index_t getIndexOfLeastFrequent(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES]);
public:
	Dictionary(){}
	DictionaryEntry<delta_t, confidence_t> read(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], bool useIndex, index_t index, delta_t delta, index_t &resultIndex);
	DictionaryEntry<delta_t, confidence_t> write(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], delta_t delta, index_t &resultIndex);

};

template<typename index_t, typename delta_t, typename confidence_t>
void Dictionary<index_t, delta_t, confidence_t>::updateConfidence(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], index_t index) {
#pragma HLS INLINE
	loop_updateConfidence: for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS UNROLL
		if (i == index) {
			if (dictionaryEntries[i].confidence < (DICTIONARY_LFU_MAX_CONFIDENCE - DICTIONARY_LFU_CONFIDENCE_STEP))
				dictionaryEntries[i].confidence += DICTIONARY_LFU_CONFIDENCE_STEP;
			else
				dictionaryEntries[i].confidence = DICTIONARY_LFU_MAX_CONFIDENCE;
		}	
		else {
			if (dictionaryEntries[i].confidence > 1)
				dictionaryEntries[i].confidence--;
			else
				dictionaryEntries[i].confidence = 0;
		}
	}
}

template<typename index_t, typename delta_t, typename confidence_t>
index_t Dictionary<index_t, delta_t, confidence_t>::getIndexOfDelta(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], delta_t delta) {
#pragma HLS INLINE
	index_t res = NUM_CLASSES;

	loop_getIndexOfDelta: for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS UNROLL
		if (dictionaryEntries[i].delta == delta) {
			res = i;
			break;
		}
	}
	return res;
}

template<typename index_t, typename delta_t, typename confidence_t>
index_t Dictionary<index_t, delta_t, confidence_t>::getIndexOfLeastFrequent(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES]) {
#pragma HLS INLINE
	index_t res = NUM_CLASSES;
	confidence_t minConfidence = DICTIONARY_LFU_MAX_CONFIDENCE;
	loop_getIndexOfLeastFrequent :for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS UNROLL
		if (dictionaryEntries[i].confidence < minConfidence) {
			res = i;
			minConfidence = dictionaryEntries[i].confidence;
		}
	}
	return res;
}

template<typename index_t, typename delta_t, typename confidence_t>
DictionaryEntry<delta_t, confidence_t> Dictionary<index_t, delta_t, confidence_t>::read(
		DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], bool useIndex, index_t index, delta_t delta, index_t &resultIndex) {
// #pragma HLS INLINE
	#pragma HLS PIPELINE
	DictionaryEntry<delta_t, confidence_t> res;

	if (useIndex) {
		resultIndex = index;
		res = dictionaryEntries[resultIndex];
	}
	else {
		resultIndex = this->getIndexOfDelta(dictionaryEntries, delta);
		if (resultIndex < NUM_CLASSES) {
			res = dictionaryEntries[resultIndex];
		}

	}
	this->updateConfidence(dictionaryEntries, resultIndex);

	return res;
}

template<typename index_t, typename delta_t, typename confidence_t>
DictionaryEntry<delta_t, confidence_t> Dictionary<index_t, delta_t, confidence_t>::write(
		DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], delta_t delta, index_t &resultIndex) {
// #pragma HLS INLINE
	#pragma HLS PIPELINE
	DictionaryEntry<delta_t, confidence_t> res;
	

	resultIndex = this->getIndexOfDelta(dictionaryEntries, delta);
	if (resultIndex < NUM_CLASSES) {
		res = dictionaryEntries[resultIndex];
		this->updateConfidence(dictionaryEntries, resultIndex);
	}
	else {
		resultIndex = this->getIndexOfLeastFrequent(dictionaryEntries);
		res.delta = delta;
		res.valid = true;
		res.confidence = DICTIONARY_LFU_INITIAL_CONFIDENCE;
		dictionaryEntries[resultIndex] = res;
		// this->updateConfidence(dictionaryEntries, resultIndex);
	}


	return res;
}

