#pragma once
#include "global.hpp"

template<typename address_t>
struct CacheFetchSnifferEntry {
    address_t address;
    bool valid;
    CacheFetchSnifferEntry() : address(0), valid(false) {}
};

template<typename address_t>
struct CacheFetchSnifferQueue {
		CacheFetchSnifferEntry<address_t> entries[CFS_QUEUE_LENGTH];
		CacheFetchSnifferQueue(){}
};

template<typename address_t, typename queue_length_t>
class CacheFetchSniffer {

    public:
    void operator()( 
    	CacheFetchSnifferEntry<address_t> queue[CFS_QUEUE_LENGTH],
        address_t inputAddress, bool inputNop,
        address_t outputAddress, bool outputNop,
        address_t& realOutputAddress, bool& realOutputNop
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
            realOutputAddress = queue[k].address; // Return the duplicate address
        
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
