#include <iostream>
#include "../include/global.hpp"

DictionaryEntry<delta_t, dic_confidence_t> testDictionary(dic_index_t index){
#pragma HLS TOP

	static DictionaryEntry<delta_t, dic_confidence_t> dictionaryEntries[NUM_CLASSES];
#pragma HLS ARRAY_PARTITION variable=dictionaryEntries dim=1 complete

	static Dictionary<dic_index_t, delta_t, dic_confidence_t> dictionary;

	dic_index_t resultingIndex = 0;
	DictionaryEntry<delta_t, dic_confidence_t> res = dictionary.write(dictionaryEntries, -2, resultingIndex);
	res = dictionary.read(dictionaryEntries, true, 0, 0, resultIndex);
	res = dictionary.read(dictionaryEntries, false, 0, -5, resultIndex);
	return res;
}
