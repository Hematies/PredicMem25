#pragma once


constexpr unsigned bitsNeeded(unsigned n) {
	return n <= 1 ? 0 : 1 + bitsNeeded((n + 1) / 2);
}

template<typename data_t, unsigned N>
struct LookUpTable {
	data_t table[N];
};

template<typename data_t, unsigned N>
constexpr LookUpTable<data_t, N> fillUniformLookUpTable(int min, int max) {
	LookUpTable<data_t, N> res;
	int numEntriesPerValue = N / (max - min + 1);
	int numRemaindingEntries = N % (max - min + 1);

	int entryIndex = 0;
	for (int value = min; value <= max; value++) {
		int numEntriesForValue = numEntriesPerValue;
		if (value == min) numEntriesForValue += numRemaindingEntries;

		for (int offset = 0; offset < numEntriesForValue; offset++) {
			res.table[entryIndex + offset] = value;
		}
		entryIndex += numEntriesForValue;
	}
	return res;
}