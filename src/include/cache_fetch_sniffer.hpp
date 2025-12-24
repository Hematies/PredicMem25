#pragma once
#include "global.hpp"

template<typename block_block_address_t>
struct CacheFetchSnifferEntry {
    block_address_t address;
    bool valid;
    CacheFetchSnifferEntry() : address(0), valid(false) {}
};

template<typename block_address_t>
struct CacheFetchSnifferQueue {
		CacheFetchSnifferEntry<block_address_t> entries[CFS_QUEUE_LENGTH];
		CacheFetchSnifferQueue(){}
};

template<typename block_address_t, typename queue_length_t>
class CacheFetchSniffer {

    public:
    void operator()( 
    	CacheFetchSnifferEntry<block_address_t> queue[CFS_QUEUE_LENGTH],
        block_address_t inputAddress, bool inputNop,
        block_address_t outputAddress, bool outputNop,
        block_address_t& realOutputAddress, bool& realOutputNop
    ) {
#pragma HLS INLINE
        realOutputAddress = 0;
        realOutputNop = true;

        // Check for duplicates in the queue
        queue_length_t k = 0;
        if(!outputNop){
            for (int i = CFS_QUEUE_LENGTH - 1; i >= 0; i--) {
                #pragma HLS UNROLL
                if (queue[i].valid &&
                    queue[i].address == outputAddress) {
                    k = i;
                    realOutputNop = false;
                    break;
                }
            }
            for (int i = CFS_QUEUE_LENGTH - 1; i >= 0; i--) {
                #pragma HLS UNROLL
                if(!realOutputNop && i >= k){
                    queue[i].valid = false; // Invalidate entries after the duplicate
                }
            }
        }
        if (!realOutputNop)
            realOutputAddress = outputAddress; // Return the duplicate address
        
        // Shift the queue if a new input is provided
        if (!inputNop) {
            for (int i = CFS_QUEUE_LENGTH - 1; i >= 1; i--) {
                #pragma HLS UNROLL
                queue[i] = queue[i - 1];

            }
            // Insert the new address
            queue[0].address = inputAddress;
            queue[0].valid = true;
        }
    }
};
