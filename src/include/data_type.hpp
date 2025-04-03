#pragma once

#include "config.hpp"
#include "ap_int.h"

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
