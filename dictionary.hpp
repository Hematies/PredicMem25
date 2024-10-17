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
	DictionaryEntry<delta_t, confidence_t> dictionaryEntries[NUM_CLASSES];
	void updateConfidence(index_t index);
	index_t getIndexOfDelta(delta_t delta);
	index_t getIndexOfLeastFrequent();
public:
	DictionaryEntry<delta_t, confidence_t> operator()(bool opRead, bool useIndex, index_t index, delta_t delta);
};



template<typename index_t, typename delta_t, typename confidence_t>
void Dictionary<index_t, delta_t, confidence_t>::updateConfidence(index_t index) {
	for (int i = 0; i < NUM_CLASSES; i++) {
		if (i == index) {
			if (this->dictionaryEntries[i].confidence < (DICTIONARY_LFU_MAX_CONFIDENCE - DICTIONARY_LFU_CONFIDENCE_STEP))
				this->dictionaryEntries[i].confidence += DICTIONARY_LFU_CONFIDENCE_STEP;
			else
				this->dictionaryEntries[i].confidence = DICTIONARY_LFU_MAX_CONFIDENCE;
		}	
		else {
			if (this->dictionaryEntries[i].confidence > 1)
				this->dictionaryEntries[i].confidence--;
			else
				this->dictionaryEntries[i].confidence = 0;
		}
	}
}

template<typename index_t, typename delta_t, typename confidence_t>
index_t Dictionary<index_t, delta_t, confidence_t>::getIndexOfDelta(delta_t delta) {
	index_t res = NUM_CLASSES;

	for (int i = 0; i < NUM_CLASSES; i++) {
		if (this->dictionaryEntries[i].delta == delta) {
			res = i;
			break;
		}
	}
	return res;
}

template<typename index_t, typename delta_t, typename confidence_t>
index_t Dictionary<index_t, delta_t, confidence_t>::getIndexOfLeastFrequent() {
	index_t res = NUM_CLASSES;
	confidence_t minConfidence = DICTIONARY_LFU_MAX_CONFIDENCE;
	for (int i = 0; i < NUM_CLASSES; i++) {
		if (this->dictionaryEntries[i].confidence < minConfidence) {
			res = i;
			minConfidence = this->dictionaryEntries[i].confidence;
		}
	}
	return res;
}

template<typename index_t, typename delta_t, typename confidence_t>
DictionaryEntry<delta_t, confidence_t> Dictionary<index_t, delta_t, confidence_t>::operator()(bool opRead, bool useIndex, index_t index, delta_t delta) {
	DictionaryEntry<delta_t, confidence_t> res;
	
	// If there is a read:
	index_t targetIndex;
	if (opRead) {
		if (useIndex) {
			targetIndex = index;
			res = this->dictionaryEntries[targetIndex];
		}
		else {
			targetIndex = this->getIndexOfDelta(delta);
			if (targetIndex < NUM_CLASSES) {
				res = this->dictionaryEntries[targetIndex];
			}
			
		}
	}
	// If there is a write:
	else { 

		targetIndex = this->getIndexOfDelta(delta);
		if (targetIndex < NUM_CLASSES) {
			res = this->dictionaryEntries[targetIndex];
		}
		else {
			targetIndex = this->getIndexOfLeastFrequent();
			res.delta = delta;
			res.valid = true;
			res.confidence = 0;
		}	
	}
	this->updateConfidence(targetIndex);

	return res;
}