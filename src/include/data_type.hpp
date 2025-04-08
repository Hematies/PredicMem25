#pragma once

#include "config.hpp"
#include "ap_int.h"

#ifdef CSIM_DEBUG
typedef uint64_t axi_data_t;
typedef uint64_t address_t;
typedef uint64_t region_address_t;
typedef uint64_t ib_index_t;
typedef uint64_t ib_way_t;
typedef uint64_t ib_tag_t;
typedef uint64_t ib_region_tag_t;
typedef uint64_t block_address_t;
typedef uint64_t class_t;
typedef uint64_t ib_confidence_t;
typedef uint64_t ib_lru_t;
typedef uint64_t dic_index_t;
typedef int64_t delta_t;
typedef uint64_t dic_confidence_t;
typedef int64_t svm_weight_t;
typedef int64_t svm_distance_t;
typedef uint64_t forwarding_index_t;
typedef uint64_t conf_forwarding_index_t;
#else
typedef ap_uint<AXI_DATA_SIZE_BITS> axi_data_t;
typedef ap_uint<NUM_ADDRESS_BITS> address_t;
typedef ap_uint<NUM_REGION_ADDRESS_BITS> region_address_t;
typedef ap_uint<IB_NUM_SETS_LOG2> ib_index_t;
typedef ap_uint<IB_NUM_WAYS_INCLUDING_NULL_LOG2> ib_way_t;
typedef ap_uint<IB_NUM_TAG_BITS> ib_tag_t;
typedef ap_uint<IB_NUM_REGION_TAG_BITS> ib_region_tag_t;
typedef ap_uint<NUM_BLOCK_ADDRESS_BITS> block_address_t;
typedef ap_uint<NUM_CLASSES_INCLUDING_NULL_LOG2> class_t;
typedef ap_uint<MAX_PREDICTION_CONFIDENCE_LOG2> ib_confidence_t;
typedef ap_uint<IB_NUM_LRU_COUNTER_BITS> ib_lru_t;
typedef ap_uint<NUM_CLASSES_LOG2> dic_index_t;
typedef ap_int<NUM_DELTA_BITS> delta_t;
typedef ap_uint<DICTIONARY_LFU_CONFIDENCE_LOG2> dic_confidence_t;
typedef ap_int<NUM_SVM_WEIGHT_BITS> svm_weight_t;
typedef ap_int<NUM_DISTANCE_BITS> svm_distance_t;
typedef ap_uint<FORWARDING_DEPTH_LOG2> forwarding_index_t;
typedef ap_uint<CONF_FORWARDING_DEPTH_LOG2> conf_forwarding_index_t;
#endif
typedef ap_uint<AXI_BURST_LENGTH_LOG2> burst_length_t;
typedef ap_uint<AXI_BURST_SIZE_BITFIELD_LOG2> burst_size_t;
typedef ap_uint<AXI_BURST_LENGTH_LOG2> burst_length_t;
typedef ap_uint<AXI_BURST_LENGTH_LOG2 + AXI_BURST_SIZE_LOG2> burst_size_and_length_t;
typedef ap_uint<AXI_BURST_BLOCK_LOG2> block_burst_length_t;
typedef ap_uint<AXI_MAX_BURST_BLOCK_LOG2> prefetch_block_burst_length_t;
typedef ap_uint<AXI_MAX_BURST_BLOCK_LOG2 + BLOCK_SIZE_LOG2> burst_length_in_words_t;

