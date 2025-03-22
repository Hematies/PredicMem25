#include <iostream>
#include "../include/global.hpp"
#include <hls_stream.h>

DictionaryEntry<delta_t, dic_confidence_t> operateDictionary(dic_index_t index, delta_t delta, bool performRead, dic_index_t &resultIndex, bool &isHit);

InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> operateInputBuffer(address_t addr, InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> entry,
		bool performRead, bool& isHit);


void operateSVM(class_t input[SEQUENCE_LENGTH], class_t target, class_t output[MAX_PREFETCHING_DEGREE]);


void prefetchWithGASP(address_t instructionPointer, block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]
		);

void prefetchWithSGASP(block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]
		);

void prefetchWithSGASPWithAXI(axi_data_t *inputAddress,
		axi_data_t *outputAddress);

void prefetchWithGASPWithNop(address_t instructionPointer, block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE], bool nop
		);

void prefetchWithSGASPWithNop(block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE], bool nop
		);


