# SPP (Signature Path Prefetcher) - HLS Implementation

## Overview

This is a hardware synthesizable (HLS) implementation of the SPP prefetcher from ChampSim. SPP is a state-of-the-art prefetching algorithm that uses signature-based pattern correlation to predict future memory accesses.

## Architecture

The SPP implementation is modular and consists of four main components:

### 1. **Signature Table (spp_signature_table.hpp)**
- Stores per-page information for each accessed memory page
- Tracks the current signature and last cache line offset within the page
- Uses LRU replacement policy with 1 set and 256 ways
- Calculates deltas between consecutive accesses to the same page
- Updates signatures using XOR-based hash with 7-bit sign-magnitude delta representation

**Key Parameters:**
- `SPP_ST_SET = 1`: Single set (associative structure)
- `SPP_ST_WAY = 256`: 256-way associativity
- `SPP_ST_TAG_BIT = 16`: 16-bit page tag
- `SPP_SIG_BIT = 12`: 12-bit signature width

### 2. **Pattern Table (spp_pattern_table.hpp)**
- Correlates signatures with cache line deltas
- Stores confidence values for each delta pattern
- Maintains local and global confidence counters
- Generates prefetch candidates based on signature and current confidence
- Implements lookahead mechanism for speculative prefetching

**Key Parameters:**
- `SPP_PT_SET = 512`: 512 sets
- `SPP_PT_WAY = 4`: 4-way associativity
- `SPP_C_SIG_BIT = 4`: 4-bit signature confidence (0-15)
- `SPP_C_DELTA_BIT = 4`: 4-bit delta confidence (0-15)

### 3. **Prefetch Filter (spp_prefetch_filter.hpp)**
- Prevents duplicate prefetch requests using hash-based filtering
- Tracks whether prefetches were actually useful
- Maintains global accuracy counters (issued vs. useful prefetches)
- Distinguishes between L2 prefetches (high confidence) and LLC prefetches (medium confidence)

**Key Parameters:**
- `SPP_FILTER_SET = 1024`: 1024-entry filter
- `SPP_QUOTIENT_BIT = 10`: Index bits for filter
- `SPP_REMAINDER_BIT = 6`: Tag bits for filter
- `SPP_FILL_THRESHOLD = 90`: Confidence threshold for L2 prefetch
- `SPP_PF_THRESHOLD = 25`: Minimum confidence for any prefetch

### 4. **Global History Register (spp_global_register.hpp)**
- Stores information about cross-page prefetch requests
- Bootstraps learning when accessing new pages
- Maintains global prefetching accuracy metric
- 8 entries, replacement based on lowest confidence

**Key Parameters:**
- `SPP_MAX_GHR_ENTRY = 8`: Maximum GHR entries
- `SPP_GLOBAL_COUNTER_MAX = 1023`: Saturation value for counters

### 5. **Main SPP Prefetcher (spp.hpp)**
- Integrates all components into a unified interface
- Implements the main prefetch generation pipeline
- Manages lookahead speculative prefetching
- Enforces page boundary protection (prefetches within same page only)

## Algorithm Flow

### Main Processing Pipeline

1. **Stage 1: Signature Table Lookup**
   - Hash page address to find signature table entry
   - Calculate delta between current and last offset
   - Update signature using XOR and delta information

2. **Stage 2: Pattern Update**
   - If previous signature exists, update pattern table with (sig, delta) pair
   - Increment confidence counters for observed patterns

3. **Stage 3: Prefetch Generation**
   - Query pattern table with current signature
   - Generate prefetch candidates meeting confidence thresholds
   - Check filter to avoid duplicates
   - Issue prefetches (L2 or LLC based on confidence)
   - Update global counters

4. **Stage 4: Lookahead**
   - For high-confidence prefetches, speculatively follow pattern chain
   - Calculate next signature and generate additional prefetches
   - Continue up to 3 lookahead levels
   - Lookahead confidence damped by global accuracy

### Confidence Calculation

For direct prefetches (depth=0):
```
local_conf = 100 * c_delta[set][way] / c_sig[set]
pf_conf = local_conf
```

For speculative prefetches (depth>0):
```
pf_conf = (global_accuracy * c_delta[set][way] / c_sig[set]) * 
          (lookahead_conf / 100)
```

## Data Types and Bitwidths

The implementation uses HLS-friendly fixed-width types via `ap_uint` and `ap_int`:

| Component | Type | Bitwidth | Purpose |
|-----------|------|----------|---------|
| Address | `address_t` | 32 | Memory address |
| Block Address | `block_address_t` | 26 | Cache-line-indexed address |
| ST Tag | `spp_st_tag_t` | 16 | Page identifier |
| Signature | `spp_st_sig_t` | 12 | Pattern signature |
| ST Confidence | `spp_st_confidence_t` | 4 | Confidence counter |
| PT Delta | `spp_pt_delta_t` | 7 | Signed offset delta |
| PT Confidence | `spp_pt_confidence_t` | 4 | Delta confidence counter |
| Filter Tag | `spp_filter_tag_t` | 6 | Filter remainder tag |
| Global Counters | - | 10 | Saturating counters |

## Configuration Parameters

All parameters are defined in `spp_config.hpp`:

```cpp
// Functional knobs
constexpr bool SPP_LOOKAHEAD_ON = true;      // Enable speculative prefetching
constexpr bool SPP_FILTER_ON = true;         // Enable duplicate filtering
constexpr bool SPP_GHR_ON = true;            // Enable global history register

// Memory hierarchy
constexpr uint32_t SPP_LOG2_PAGE_SIZE = 12;  // 4KB pages
constexpr uint32_t SPP_LOG2_BLOCK_SIZE = 6;  // 64B cache lines
```

## Usage Example

```cpp
// Declare SPP instance
SPPPrefetcher spp;

// Arrays for prefetch output
ap_int<7> prefetch_deltas[SPP_MAX_PREFETCH_QUEUE];
ap_uint<4> prefetch_confidences[SPP_MAX_PREFETCH_QUEUE];
uint32_t num_prefetches, num_l2_prefetches;

// Process cache access
spp.process_cache_access(
    address,           // Input: cache line address
    cache_hit,         // Input: hit/miss from L2
    useful_prefetch,   // Input: feedback from fill
    prefetch_deltas,   // Output: prefetch offsets
    prefetch_confidences, // Output: confidence values
    num_prefetches,    // Output: total prefetches
    num_l2_prefetches  // Output: L2-level prefetches
);

// Handle cache evictions
spp.notify_cache_evict(evicted_address);

// Handle demand hits on prefetched data
spp.notify_cache_hit(hit_address);
```

## HLS Optimization Pragmas

The implementation uses the following HLS pragmas for optimization:

- `#pragma HLS PIPELINE`: Enables loop pipelining for performance
- `#pragma HLS UNROLL`: Fully unrolls small loops for parallelism
- `#pragma HLS UNROLL collapse=2`: Unrolls nested loops
- `#pragma HLS ARRAY_PARTITION`: Partitions arrays for parallel access

## Key Design Decisions

1. **Template-based Architecture**: Uses C++ templates for type flexibility while maintaining hardware efficiency via `ap_uint` specialization.

2. **Modular Components**: Each major structure (ST, PT, Filter, GHR) is independent and reusable.

3. **Page Boundary Protection**: Prefetches strictly stay within the same page to avoid speculative faults.

4. **Confidence-based Throttling**: Multiple confidence thresholds allow fine-grained control over prefetch aggressiveness.

5. **Global Accuracy Tracking**: Maintains running statistics of prefetch effectiveness for dynamic adaptation.

## Integration with PredicMem25

The SPP implementation integrates with the PredicMem25 project through:

- **Data Types**: Uses PredicMem25's `address_t`, `block_address_t`, and config parameters
- **Memory Configuration**: Respects `BLOCK_SIZE_LOG2` and memory hierarchy settings
- **Initialization**: Provides `spp_init.hpp` with structured data matrices

## Files Included

1. **spp_config.hpp** - Configuration constants and parameters
2. **spp_global_register.hpp** - Global History Register implementation
3. **spp_signature_table.hpp** - Signature Table implementation
4. **spp_pattern_table.hpp** - Pattern Table implementation
5. **spp_prefetch_filter.hpp** - Prefetch Filter implementation
6. **spp.hpp** - Main SPP prefetcher class
7. **spp_init.hpp** - Initialization functions and type definitions

## References

- Original SPP Paper: "Signature Path Prefetcher" (MICRO 2015)
- ChampSim SPP Reference: https://github.com/ChampSim/ChampSim/tree/master/prefetcher_code/spp

## Notes

- All components are synthesizable for FPGA/ASIC implementation
- The implementation is fully parameterized for easy extension
- Confidence values use 4-bit fixed-point representations for efficiency
- Memory access patterns are not forwarded across cache boundaries
