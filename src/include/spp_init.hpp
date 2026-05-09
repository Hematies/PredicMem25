#pragma once

#include "spp.hpp"
#include "spp_config.hpp"
#include "spp_data_type.hpp"

// ============================================================================
// SPP Initialization and Type Definitions
// ============================================================================
// Provides convenient type aliases and initialization functions
// for the SPP prefetcher in the PredicMem25 framework
//
// All types are defined in spp_data_type.hpp and used as default
// template parameters in SPP components

// Full SPP instance type with default parameters
typedef SPP<spp_address_t, spp_block_address_t, spp_st_tag_t,
            spp_st_sig_t, spp_st_confidence_t,
            spp_pt_delta_t, spp_pt_confidence_t, spp_filter_tag_t> SPPPrefetcher;

// ============================================================================
// SPP Data Structure Matrices (for memory layout in HLS)
// ============================================================================

// Signature Table storage matrix
struct SPPSignatureTableMatrix {
    ap_uint<1> valid[SPP_ST_SET][SPP_ST_WAY];
    spp_st_tag_t tag[SPP_ST_SET][SPP_ST_WAY];
    spp_page_offset_t last_offset[SPP_ST_SET][SPP_ST_WAY];
    spp_st_sig_t sig[SPP_ST_SET][SPP_ST_WAY];
    spp_st_lru_t lru[SPP_ST_SET][SPP_ST_WAY];
};

// Pattern Table storage matrix
struct SPPPatternTableMatrix {
    spp_pt_delta_t delta[SPP_PT_SET][SPP_PT_WAY];
    spp_pt_confidence_t c_delta[SPP_PT_SET][SPP_PT_WAY];
    spp_pt_confidence_t c_sig[SPP_PT_SET];
};

// Prefetch Filter storage matrix
struct SPPPrefetchFilterMatrix {
    spp_filter_tag_t remainder_tag[SPP_FILTER_SET];
    ap_uint<1> valid[SPP_FILTER_SET];
    ap_uint<1> useful[SPP_FILTER_SET];
};

// Global Register storage
struct SPPGlobalRegisterStorage {
    spp_ghr_counter_t pf_issued;
    spp_ghr_counter_t pf_useful;
    spp_accuracy_t global_accuracy;
    
    spp_ghr_valid_t valid[SPP_MAX_GHR_ENTRY];
    spp_st_sig_t sig[SPP_MAX_GHR_ENTRY];
    spp_st_confidence_t confidence[SPP_MAX_GHR_ENTRY];
    spp_ghr_offset_t offset[SPP_MAX_GHR_ENTRY];
    spp_pt_delta_t delta[SPP_MAX_GHR_ENTRY];
};

// ============================================================================
// Initialization Helper Functions
// ============================================================================

// Initialize SPP Signature Table
constexpr SPPSignatureTableMatrix initSPPSignatureTable() {
    SPPSignatureTableMatrix res;
    for (int set = 0; set < SPP_ST_SET; set++) {
        for (int way = 0; way < SPP_ST_WAY; way++) {
            res.valid[set][way] = 0;
            res.tag[set][way] = 0;
            res.last_offset[set][way] = 0;
            res.sig[set][way] = 0;
            res.lru[set][way] = way;
        }
    }
    return res;
}

// Initialize SPP Pattern Table
constexpr SPPPatternTableMatrix initSPPPatternTable() {
    SPPPatternTableMatrix res;
    for (int set = 0; set < SPP_PT_SET; set++) {
        for (int way = 0; way < SPP_PT_WAY; way++) {
            res.delta[set][way] = 0;
            res.c_delta[set][way] = 0;
        }
        res.c_sig[set] = 0;
    }
    return res;
}

// Initialize SPP Prefetch Filter
constexpr SPPPrefetchFilterMatrix initSPPPrefetchFilter() {
    SPPPrefetchFilterMatrix res;
    for (int set = 0; set < SPP_FILTER_SET; set++) {
        res.remainder_tag[set] = 0;
        res.valid[set] = 0;
        res.useful[set] = 0;
    }
    return res;
}

// Initialize SPP Global Register
constexpr SPPGlobalRegisterStorage initSPPGlobalRegister() {
    SPPGlobalRegisterStorage res;
    res.pf_issued = 0;
    res.pf_useful = 0;
    res.global_accuracy = 0;

    for (int i = 0; i < SPP_MAX_GHR_ENTRY; i++) {
        res.valid[i] = 0;
        res.sig[i] = 0;
        res.confidence[i] = 0;
        res.offset[i] = 0;
        res.delta[i] = 0;
    }
    return res;
}

// ============================================================================
// SPP Prefetch Output Structure
// ============================================================================

struct SPPPrefetchOutput {
    spp_pt_delta_t deltas[SPP_MAX_PREFETCH_QUEUE];
    spp_pt_confidence_t confidences[SPP_MAX_PREFETCH_QUEUE];
    uint32_t num_prefetches;
    uint32_t num_l2_prefetches;
    bool valid;
};

// ============================================================================
// Static Initialization Helpers
// ============================================================================
// Usage example for static initialization in HLS:
//
// static SPPSignatureTableMatrix sppSignatureTableMatrix = 
//     initSPPSignatureTable();
// #pragma HLS ARRAY_RESHAPE variable=sppSignatureTableMatrix.sig complete
// #pragma HLS ARRAY_RESHAPE variable=sppSignatureTableMatrix.tag complete
//
// static SPPPatternTableMatrix sppPatternTableMatrix = 
//     initSPPPatternTable();
// #pragma HLS ARRAY_PARTITION variable=sppPatternTableMatrix.delta complete
//
// static SPPPrefetchFilterMatrix sppPrefetchFilterMatrix = 
//     initSPPPrefetchFilter();
// #pragma HLS ARRAY_PARTITION variable=sppPrefetchFilterMatrix.remainder_tag complete
//
// static SPPGlobalRegisterStorage sppGlobalRegisterStorage = 
//     initSPPGlobalRegister();
