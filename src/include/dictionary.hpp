#pragma once
#include "global.hpp"

template<typename delta_t, typename confidence_t>
struct DictionaryEntry {
	bool valid;
	delta_t delta;
	confidence_t confidence;
};


template<typename delta_t, typename confidence_t>
struct DictionaryEntriesMatrix {
	DictionaryEntry<delta_t, confidence_t> entries[NUM_CLASSES];
	DictionaryEntriesMatrix(){}
};


template<typename index_t, typename delta_t, typename confidence_t> 
class Dictionary {
protected:
	void updateConfidence(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], index_t index);
	index_t getIndexOfDelta(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], delta_t delta);
	index_t getIndexOfLeastFrequent(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES]);
public:
	Dictionary(){}
	DictionaryEntry<delta_t, confidence_t> read(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], bool useIndex, index_t index, delta_t delta, index_t &resultIndex,
			bool performUpdateConfidence, bool &isHit);
	DictionaryEntry<delta_t, confidence_t> write(DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], delta_t delta, index_t &resultIndex, bool &isHit);

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

	index_t indexes[NUM_CLASSES / 2];
	confidence_t minConfidences[NUM_CLASSES / 2];
	for (int i = 0; i < NUM_CLASSES; i+=2) {
		#pragma HLS UNROLL
		if(i != (NUM_CLASSES - 1)){
			bool lower = dictionaryEntries[i].confidence < dictionaryEntries[i+1].confidence;
			indexes[i >> 1] = lower? i : i+1;
			minConfidences[i >> 1] = lower? dictionaryEntries[i].confidence : dictionaryEntries[i+1].confidence;
		}

	}
	index_t res = indexes[0];
	confidence_t minConfidence = minConfidences[0];
	for (int i = 0; i < NUM_CLASSES / 2; i++) {
		#pragma HLS UNROLL
		if (minConfidences[i] < minConfidence) {
			res = indexes[i];
			minConfidence = minConfidences[i];
		}
	}
	if(NUM_CLASSES % 2 == 1){
		if (dictionaryEntries[NUM_CLASSES - 1].confidence < minConfidence) {
			res = NUM_CLASSES - 1;
		}
	}


	return res;
}

template<typename index_t, typename delta_t, typename confidence_t>
DictionaryEntry<delta_t, confidence_t> Dictionary<index_t, delta_t, confidence_t>::read(
		DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], bool useIndex, index_t index, delta_t delta, index_t &resultIndex,
		bool performUpdateConfidence, bool &isHit) {
	#pragma HLS PIPELINE
	DictionaryEntry<delta_t, confidence_t> res;
	isHit = true;

	if (useIndex) {
		resultIndex = index;
		res = dictionaryEntries[resultIndex];
	}
	else {
		resultIndex = this->getIndexOfDelta(dictionaryEntries, delta);
		isHit = resultIndex < NUM_CLASSES;
		if (resultIndex < NUM_CLASSES) {
			res = dictionaryEntries[resultIndex];
		}

	}
	if(performUpdateConfidence)
		this->updateConfidence(dictionaryEntries, resultIndex);

	return res;
}

template<typename index_t, typename delta_t, typename confidence_t>
DictionaryEntry<delta_t, confidence_t> Dictionary<index_t, delta_t, confidence_t>::write(
		DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES], delta_t delta, index_t &resultIndex, bool &isHit) {
#pragma HLS ARRAY_PARTITION variable=dictionaryEntries complete

	#pragma HLS INLINE

	DictionaryEntry<delta_t, confidence_t> res;
	
	index_t leastFrequentIndex = this->getIndexOfLeastFrequent(dictionaryEntries);
	resultIndex = this->getIndexOfDelta(dictionaryEntries, delta);
	isHit = resultIndex < NUM_CLASSES;
	if (resultIndex < NUM_CLASSES) {
		res = dictionaryEntries[resultIndex];
	}
	else {
		resultIndex = leastFrequentIndex;
		res.delta = delta;
		res.valid = true;
		res.confidence = DICTIONARY_LFU_INITIAL_CONFIDENCE;
		dictionaryEntries[(int)resultIndex] = res;
	}
	this->updateConfidence(dictionaryEntries, resultIndex);


	return res;
}

