#include <iostream>
#include "../include/global.hpp"
#include <hls_stream.h>
#include "hls_burst_maxi.h"

DictionaryEntry<delta_t, dic_confidence_t> operateDictionary(dic_index_t index, delta_t delta, bool performRead, dic_index_t &resultIndex, bool &isHit);

InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_lru_t> operateInputBuffer(address_t addr, InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_lru_t> entry,
		bool performRead, bool& isHit);


void operateSVM(class_t input[SEQUENCE_LENGTH], class_t target, class_t output[MAX_PREFETCHING_DEGREE]);

void operateSVMWithNop(class_t input[SEQUENCE_LENGTH], class_t target, class_t output[MAX_PREFETCHING_DEGREE],
		bool nop);

void prefetchWithGASP(address_t instructionPointer, block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]
		);

void prefetchWithSGASP(block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]
		);

void prefetchWithSGASPWithAXI(address_t inputAddress,
		axi_data_t *readPort,
		axi_data_t prefetchedData[MAX_PREFETCHING_DEGREE]
		);

void prefetchWithBSGASPWithAXI(address_t inputAddress,
		burst_size_t burstSize,
		burst_length_t burstLength,
		hls::burst_maxi<axi_data_t> readPort,
		hls::stream<axi_data_t>& prefetchedData
		);

void prefetchWithBSGASPWithDataflowWithAXI(address_t inputAddress,
		burst_size_t burstSize,
		burst_length_t burstLength,
		hls::burst_maxi<axi_data_t> readPort,
		hls::stream<axi_data_t>& prefetchedData
		);

void prefetchWithGASPWithNop(address_t instructionPointer, block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE], bool nop
		);

void prefetchWithSGASPWithNop(block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE], bool nop
		);

void prefetchWithSGASPWithNopWithDataflow(block_address_t memoryAddress,
		block_address_t& prefetchAddress,
		bool nop
		);

void prefetchWithBSGASPWithNop(address_t inputAddress,
		burst_size_t burstSize,
		burst_length_t burstLength,
		address_t& prefetchAddress,
		burst_length_in_words_t& totalBurstLength,
		bool nop
		);

void prefetchWithBSGASPWithNopWithDataflow(address_t inputAddress,
		burst_size_t burstSize,
		burst_length_t burstLength,
		address_t& prefetchAddress,
		burst_length_in_words_t& totalBurstLength,
		bool nop
		);

void prefetchWithBSGASPWithNopWithDataflowForTesting(block_address_t memoryBlockAddress,
	block_burst_length_t inputBlockBurstLength,
	block_address_t& prefetchAddress,
	prefetch_block_burst_length_t& outputBlockBurstLength,
	bool nop
	);
