# BOP (Best Offset Pattern) Prefetcher - HLS Implementation

## Overview
Implementation of the Best Offset Pattern prefetcher from ChampSim, optimized for Vivado HLS synthesis and integrated into the PredicMem25 framework.

## Algorithm Summary

### Best Offset Pattern (BOP)
BOP discovers recurring access patterns by testing candidate cache-line offsets:

1. **Recency Ring (RR)**: Maintains a FIFO buffer of recently accessed cache line addresses
2. **Candidate Testing**: On each cache access, tests one candidate offset (round-robin)
   - Checks if (address - candidate_offset) exists in recency ring
   - If match found, increments score for that offset
3. **Learning Phases**: 
   - Phase ends when: `round_counter >= BOP_MAX_ROUNDS` (100) OR `max_score >= BOP_MAX_SCORE` (31)
   - Selects top-N best-performing offsets based on scores
   - Resets scores and begins new phase
4. **Prefetch Generation**: Issues prefetches for best discovered offsets

### Key Parameters
- **Recency Ring Size**: 256 entries (64-byte cache lines)
- **Candidate Offsets**: 44 offsets tested: {±1, ±2, ±3, ..., ±40}
- **Max Learning Rounds**: 100 accesses per phase
- **Score Saturation**: 31 (triggers phase end if any offset reaches this)
- **Top-N Selection**: 1 offset selected (BOP_TOP_N=1)
- **Prefetch Degree**: 1 per cycle (HLS-optimized, configurable via BOP_PREF_DEGREE)

## File Structure

### Header Files (7 total)

#### 1. **bop_config.hpp**
Central configuration using `#define` constants (following SPP/GASP pattern):
```cpp
#define BOP_RR_SIZE 256              // Recency ring buffer size
#define BOP_NUM_CANDIDATES 44        // Offset candidates to test
#define BOP_MAX_ROUNDS 100           // Phase end threshold
#define BOP_MAX_SCORE 31             // Score saturation threshold
#define BOP_TOP_N 1                  // Best offsets to keep
#define BOP_SINGLE_PREFETCH 1        // Single prefetch per cycle (HLS optimization)
```

#### 2. **bop_data_type.hpp**
Centralized type definitions with dual compilation modes:

**CSIM_DEBUG mode** (simulation):
- `bop_address_t` = uint64_t
- `bop_block_address_t` = uint32_t
- `bop_candidate_t` = int8_t (signed for negative offsets)
- `bop_score_t` = uint8_t

**HLS mode** (Vivado synthesis):
- `bop_address_t` = ap_uint<32>
- `bop_block_address_t` = ap_uint<26>
- `bop_candidate_t` = ap_int<8>
- `bop_score_t` = ap_uint<6>

All types parameterized for template classes with sensible defaults.

#### 3. **bop_recency_ring.hpp**
Circular FIFO buffer for tracking recently accessed cache lines:

```cpp
template <typename rr_entry_t = bop_rr_entry_t,
          typename rr_index_t = bop_rr_index_t>
class BOPRecencyRing {
    rr_entry_t entries[BOP_RR_SIZE];
    rr_index_t head;
    
    bop_valid_t search_entry(rr_entry_t addr);    // O(BOP_RR_SIZE) unrolled
    void insert_entry(rr_entry_t addr);            // FIFO replacement
    void clear();
};
```

- **search_entry()**: Fully unrolled loop (II=1), O(RR_SIZE) to find address
- **insert_entry()**: FIFO replacement, head wraps at buffer size
- HLS Pragmas: UNROLL FACTOR=16 for search, circular buffer synthesis

#### 4. **bop_pattern_learner.hpp**
Tracks candidate offset effectiveness through scoring:

```cpp
template <typename candidate_t = bop_candidate_t,
          typename score_t = bop_score_t,
          typename candidate_index_t = bop_candidate_index_t>
class BOPPatternLearner {
    score_t scores[BOP_NUM_CANDIDATES];
    bop_candidate_loop_t candidate_ptr;
    bop_counter_t round_counter;
    candidate_t best_offsets[BOP_TOP_N];
    
    void increment_score();                      // Increment current candidate's score
    void next_candidate();                       // Move to next candidate (round-robin)
    bop_valid_t check_phase_end();              // Check if phase should end
    void select_best_offsets_indices(...);      // Find top-N scoring offsets
    void reset_phase();                          // Clear scores for next phase
};
```

- **increment_score()**: Saturating increment (max BOP_MAX_SCORE)
- **check_phase_end()**: Returns true if rounds exceeded OR max score reached
- **select_best_offsets_indices()**: Heap-like selection for top-N
- HLS Pragmas: UNROLL FACTOR=8 for score scanning

#### 5. **bop_prefetch_buffer.hpp**
Optional prefetch queue for decoupling generation and issuance:

```cpp
template <typename address_t = bop_address_t,
          typename index_t = bop_rr_index_t>
class BOPPrefetchBuffer {
    address_t buffer[BOP_PREF_BUFFER_SIZE];
    index_t head, tail, count;
    
    bop_valid_t is_full();
    bop_valid_t is_empty();
    bop_valid_t buffer_prefetch(address_t addr);
    address_t pop_prefetch();
};
```

- Typically disabled in HLS mode (BOP_ENABLE_PREF_BUFFER=0)
- FIFO circular buffer implementation
- Supports prefetch throttling if needed

#### 6. **bop.hpp** - Main Prefetcher Class
Integrates all components following PredicMem25 framework:

```cpp
template <typename address_t = bop_address_t,
          typename block_address_t = bop_block_address_t,
          /* 6 template parameters with defaults */
class BOPrefetcher {
    BOPRecencyRing<rr_entry_t> recency_ring;
    BOPPatternLearner<candidate_t, score_t, candidate_index_t> pattern_learner;
    BOPPrefetchBuffer<address_t> prefetch_buffer;
    
    void process_cache_access(address_t addr,
                              bop_valid_t cache_hit,
                              bop_valid_t useful_prefetch,
                              bop_offset_t* prefetch_deltas,
                              score_t* prefetch_confidences,
                              uint32_t& num_prefetches,
                              uint32_t& num_prefetches_l2);
    
    void notify_cache_fill(address_t filled_addr);
    void clear();
};
```

##### Main Pipeline: process_cache_access()

**HLS-Optimized for Single-Cycle Throughput (II=1)**

Stages:
1. **Static Initialization**: Constexpr matrices loaded at method start
2. **Block Address Extraction**: Convert address to cache-line address
3. **Candidate Testing**: Check if (block_addr - candidate_offset) in RR
4. **Score Update**: Increment offset score if pattern matched
5. **RR Insertion**: Add current address to recency ring
6. **Candidate Advance**: Move to next offset (round-robin)
7. **Phase Check**: Test for phase end condition
8. **Best Offset Selection**: When phase ends, select top offsets
9. **Prefetch Generation**: Output single prefetch per cycle (BOP_SINGLE_PREFETCH=1)

**Key Optimizations**:
- Single prefetch output per cycle (not 4)
- No nested loops or data dependencies
- Static constexpr initialization (compile-time, no runtime overhead)
- Fully unrolled critical path (RR search, score selection)
- HLS Pragmas: `#pragma HLS PIPELINE II=1`, `#pragma HLS UNROLL`

#### 7. **bop_init.hpp**
Constexpr-compatible initialization structures:

```cpp
struct BOPRecencyRingMatrix { /* ... */ };
struct BOPPatternLearnerMatrix { /* ... */ };
struct BOPPrefetchBufferMatrix { /* ... */ };
struct BOPCandidateOffsets { /* hardcoded 44 offsets */ };

inline constexpr BOPRecencyRingMatrix initBOPRecencyRing();
inline constexpr BOPPatternLearnerMatrix initBOPPatternLearner();
inline constexpr BOPPrefetchBufferMatrix initBOPPrefetchBuffer();
inline constexpr BOPCandidateOffsets initBOPCandidateOffsets();
```

- All matrices initialized via constexpr functions
- No runtime overhead for static initialization
- Compile-time array generation
- Compatible with HLS array partitioning/reshaping pragmas

## HLS Synthesis Guidelines

### Recommended Pragmas for optimal synthesis:

```cpp
// In process_cache_access()
#pragma HLS PIPELINE II=1                        // Single-cycle throughput
#pragma HLS UNROLL FACTOR=16                     // Unroll RR search
#pragma HLS ARRAY_PARTITION variable=entries cyclic=16

// In recency_ring.search_entry()
#pragma HLS UNROLL                               // Full unroll small loops

// In pattern_learner methods
#pragma HLS UNROLL FACTOR=8                      // Unroll score loops
#pragma HLS ARRAY_PARTITION variable=scores complete
```

### Synthesis Results (Expected)
- **Latency**: 1 cycle (II=1)
- **Resource Usage**: ~15K-20K LUTs (depending on FPGA)
- **Clock Frequency**: 250+ MHz (typical HLS results)
- **Throughput**: 1 cache access per cycle

## Integration with PredicMem25

The BOP implementation follows the same patterns as GASP and SPP:

1. **Type System**: Uses `bop_data_type.hpp` for CSIM_DEBUG/HLS duality
2. **Configuration**: All parameters as `#define` in `bop_config.hpp`
3. **Initialization**: Constexpr static matrices in `bop_init.hpp`
4. **Framework Compatibility**: Signature matches SPP/GASP:
   ```cpp
   void process_cache_access(address_t addr,
                             bop_valid_t cache_hit,
                             bop_valid_t useful_prefetch,
                             bop_offset_t* prefetch_deltas,
                             score_t* prefetch_confidences,
                             uint32_t& num_prefetches,
                             uint32_t& num_prefetches_l2);
   ```

## Usage Example

```cpp
// Create prefetcher instance (default template parameters)
BOPrefetcher<> bop;

// On each cache access
bop.process_cache_access(addr, cache_hit, useful_pf,
                         prefetch_deltas, confidences,
                         num_prefetches, num_prefetches_l2);

// On cache fill (insert backward predictions)
bop.notify_cache_fill(filled_addr);

// Reset prefetcher state
bop.clear();
```

## Comparison with Original ChampSim BOP

| Feature | Original | HLS Implementation |
|---------|----------|-------------------|
| Prefetch Degree | Configurable (1-4) | 1 per cycle (BOP_SINGLE_PREFETCH=1) |
| Lookahead | Variable | Fixed, unrolled |
| Initialization | Runtime constructors | Compile-time constexpr |
| Data Types | C++ native | ap_int fixed-width |
| Pipeline Depth | Variable | 1 cycle (II=1) |
| Synthesis | N/A | Vivado HLS compatible |

## Future Enhancements

1. **Adaptive Phase Length**: Dynamic adjustment based on memory patterns
2. **Selective Prefetching**: Confidence-based filtering
3. **Multi-Stream Support**: Per-stream pattern tracking
4. **DRAM Bandwidth Throttling**: Adapt prefetch rate to memory utilization
5. **Scoring Variants**: Alternative scoring metrics (latency-aware, power-aware)

## References

- Original ChampSim BOP: `champsim_impls/bop/`
- PredicMem25 Framework: GASP/SPP implementations
- HLS Best Practices: Vivado HLS User Guide

---
**Implementation Date**: 2025
**Framework**: Vivado HLS (2021.1+)
**Target FPGA**: Xilinx 7-series and Ultrascale
