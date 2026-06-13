# MLOP (Multi-Lookahead Offset Prefetcher) - HLS Implementation

## Overview

This directory contains the HLS (High-Level Synthesis) implementation of the MLOP prefetcher, ported from its original ChampSim implementation. MLOP learns prefetch offsets by tracking access patterns within cache zones (pages) and maintaining scores for various offsets based on their accuracy in predicting future cache accesses.

All data structures use **constexpr initialization** following the same pattern as BOP and SPP, ensuring compile-time initialization with no dynamic memory allocation.

## Algorithm Summary

### Core Concept
MLOP divides the address space into zones (typically 4KB pages) and tracks which cache lines within each zone have been accessed. It maintains confidence scores for different prefetch offsets, updating these scores when the offsets correctly predict cache hits.

### Key Stages

1. **Access Tracking**: On each cache access, MLOP:
   - Extracts the zone address and offset within zone
   - Looks up zone entry in the statically-allocated access map table
   - Updates scores for offsets based on recent access patterns

2. **Learning Phase**: After NUM_UPDATES accesses:
   - Computes maximum scores for each prefetch degree
   - Selects offsets above confidence thresholds
   - Resets scores for the next learning round

3. **Prefetch Generation**: On each access:
   - Generates prefetch requests using learned offsets
   - Filters prefetches based on the access map state (if needed)
   - Issues offsets that haven't been recently accessed or prefetched

## Compilation Modes

MLOP supports two prefetch modes, controlled by `MLOP_SINGLE_PREFETCH` in `mlop_config.hpp`:

### Single-Prefetch Mode (MLOP_SINGLE_PREFETCH = 1) - Default for HLS
- **Throughput**: Exactly 1 prefetch per cycle
- **Initiation Interval**: II=1 (optimal for hardware)
- **Selection**: Uses only the best-scoring offset from the highest degree
- **Resource Usage**: Minimal area, predictable latency
- **Code Path**: Compiled with `#if MLOP_SINGLE_PREFETCH`

**To use:**
```cpp
#define MLOP_SINGLE_PREFETCH 1   // In mlop_config.hpp
```

### Multi-Prefetch Mode (MLOP_SINGLE_PREFETCH = 0)
- **Throughput**: Up to MLOP_PF_DEGREE prefetches per cycle
- **Initiation Interval**: May require II > 1
- **Selection**: Uses multiple offsets from all degrees
- **Resource Usage**: Higher area, more complex pipelining
- **Code Path**: Compiled with `#else (when MLOP_SINGLE_PREFETCH = 0)`

**To use:**
```cpp
#define MLOP_SINGLE_PREFETCH 0   // In mlop_config.hpp
```

## Data Structures

### Offset Scores Matrix (MLOPOffsetScoresMatrix)
- **Dimensions**: [MLOP_PF_DEGREE][MLOP_NUM_OFFSETS]
- **Storage**: Statically allocated, initialized at compile-time
- **Initialization**: `initMLOPOffsetScores()` constexpr function
- **Purpose**: Tracks how many times each offset predicted a cache hit
- **Reset**: After each learning round completes

### Best Offsets Matrix (MLOPBestOffsetsMatrix)
- **Dimensions**: [MLOP_PF_DEGREE][MLOP_NUM_OFFSETS]
- **Storage**: Statically allocated with count tracking
- **Initialization**: `initMLOPBestOffsets()` constexpr function
- **Purpose**: Stores selected offsets exceeding thresholds
- **Selection**: Offsets with maximum scores above threshold per degree

### Prefetch Level Matrix (MLOPPrefetchLevelMatrix)
- **Dimensions**: [MLOP_PF_DEGREE]
- **Initialization**: `initMLOPPrefetchLevels()` constexpr function
- **Purpose**: Stores fill level for each degree

### Access Map Table (MLOPAccessMapTable)
- **Dimensions**: [MLOP_AMT_SIZE][MLOP_BLOCKS_IN_ZONE]
- **Storage**: Statically allocated with LRU metadata
- **Initialization**: `initMLOPAccessMapTable()` constexpr function
- **Purpose**: Maintains per-zone state of all blocks

## Configuration Parameters

Located in `mlop_config.hpp`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MLOP_SINGLE_PREFETCH` | 1 | Single (1) or multi (0) prefetch mode |
| `MLOP_PF_DEGREE` | 16 | Number of prefetch degrees |
| `MLOP_NUM_UPDATES` | 500 | Accesses per learning round |
| `MLOP_L1D_THRESH` | 2.0 | L1D threshold ratio |
| `MLOP_L2C_THRESH` | 0.75 | L2C threshold ratio |
| `MLOP_LLC_THRESH` | 0.3 | LLC threshold ratio |
| `MLOP_BLOCKS_IN_ZONE` | 64 | Blocks per zone (4KB / 64B) |
| `MLOP_NUM_OFFSETS` | 127 | Total offset possibilities |
| `MLOP_AMT_SIZE` | 32 | Access map table entries |

## Data Types

Located in `mlop_data_type.hpp`:

- **Simulation Mode** (CSIM_DEBUG): Uses standard C++ types for debugging
- **HLS Mode**: Uses `ap_uint`/`ap_int` for synthesis
  - Address types sized for typical cache hierarchies (32-bit)
  - Score counters can track up to NUM_UPDATES
  - Offset types are signed (support negative offsets -63 to +63)
  - All types are fixed-width for predictable HLS synthesis

## Files

- `mlop_config.hpp`: Configuration constants and derived parameters
- `mlop_data_type.hpp`: Type definitions for MLOP components
- `mlop_init.hpp`: Compile-time constexpr initialization functions and matrix structures
- `mlop.hpp`: Main MLOPrefetcher template class

## Initialization Pattern

Following BOP and SPP conventions, all MLOP data structures are initialized using constexpr functions at compile-time:

```cpp
// In mlop_init.hpp - Executed at compile-time
struct MLOPOffsetScoresMatrix {
    mlop_score_t scores[MLOP_PF_DEGREE][MLOP_NUM_OFFSETS];
    
    constexpr MLOPOffsetScoresMatrix() {
        for (int d = 0; d < MLOP_PF_DEGREE; d++) {
            for (int o = 0; o < MLOP_NUM_OFFSETS; o++) {
                scores[d][o] = 0;
            }
        }
    }
};

inline constexpr MLOPOffsetScoresMatrix initMLOPOffsetScores() {
    return MLOPOffsetScoresMatrix();
}
```

In `mlop.hpp`, these are then used as static constexpr to avoid runtime initialization:

```cpp
void process_cache_access(...) {
    #pragma HLS PIPELINE II=1
    
    // All initialization happens at compile-time
    static const MLOPOffsetScoresMatrix offset_scores = initMLOPOffsetScores();
    static const MLOPBestOffsetsMatrix best_offsets = initMLOPBestOffsets();
    static const MLOPPrefetchLevelMatrix pf_levels = initMLOPPrefetchLevels();
    static const MLOPAccessMapTable amt = initMLOPAccessMapTable();
    
    // HLS directives for array partitioning
    #pragma HLS ARRAY_PARTITION variable=offset_scores.scores complete dim=2
    #pragma HLS ARRAY_PARTITION variable=best_offsets.offsets complete dim=2
    
    // ... rest of prefetcher logic
}
```

## Integration

The MLOP HLS implementation follows the same interface pattern as other HLS prefetchers (BOP, SPP, GASP):

```cpp
// Create prefetcher instance
MLOPrefetcher<> mlop;

// Process each cache access
mlop.process_cache_access(
    address,           // Physical memory address
    cache_hit,         // Was this a hit or miss?
    useful_prefetch,   // Was previous prefetch useful?
    prefetch_deltas,   // Output: offsets to prefetch
    confidences,       // Output: confidence scores (unused)
    num_prefetches,    // Output: number of prefetches
    num_prefetches_l2  // Output: L2 prefetch count (unused)
);

// Register cache fills (optional in current implementation)
mlop.register_fill(address, set, way, prefetch_flag, evicted_addr);
```

## Porting Notes

Key differences from ChampSim implementation:

1. **Hardware Efficiency**: 
   - Constexpr initialization eliminates runtime setup overhead
   - Single prefetch per cycle (II=1) by default for optimal synthesis
   - Static arrays avoid dynamic memory management

2. **Data Structure Simplification**:
   - Access map stored as fixed arrays rather than dynamically allocated vectors
   - All structures pre-allocated at compile-time
   - Zone entries allocated statically in access map table

3. **HLS Directives**:
   - Pipeline II=1 for main processing loop
   - Array partitioning for score matrices
   - Constexpr initialization for compile-time evaluation

## Performance Characteristics

### Single-Prefetch Mode (Default)
- **Throughput**: 1 cache access processed per cycle
- **Latency**: Single cycle pipeline
- **Area**: ~2-3 KB for offset scores, access maps, and metadata
- **Estimated LUTs**: 2K-5K LUTs (varies by target device)
- **Estimated BRAMs**: 1-2 BRAM18 blocks for score matrices

### Multi-Prefetch Mode
- **Throughput**: Reduced due to increased loop complexity
- **Latency**: May be 2+ cycles per access
- **Area**: Similar storage, but more complex routing
- **Estimated LUTs**: 5K-10K LUTs

## Compilation

### For HLS (Vivado HLS, Intel HLS Compiler, etc.):
```bash
# Compile without CSIM_DEBUG flag for hardware synthesis
hls_compiler -D MLOP_HARDWARE mlop.hpp
```

### For C Simulation (with debugging):
```bash
# Compile with CSIM_DEBUG flag for functional verification
g++ -DCSIM_DEBUG -o mlop_sim test_mlop.cpp
```

## Future Enhancements

1. **Dynamic Threshold Adjustment**: Adjust thresholds based on running performance
2. **Bandwidth Throttling**: Prefetch filtering based on memory traffic
3. **Confidence Weighting**: Weight offsets by confidence rather than binary selection
4. **Streaming Integration**: Support HLS dataflow pragmas for pipelined operation
5. **State Compression**: Pack access maps more efficiently for area reduction
6. **History Queue Enhancement**: Full implementation of lookahead depth tracking

## Related Files

- Original ChampSim implementation: `champsim_impls/mlop/` (ChampSim versions)
- SPP HLS implementation: `src/include/spp.hpp` (similar constexpr pattern)
- BOP HLS implementation: `src/include/bop.hpp` (similar constexpr pattern)
- GASP HLS implementation: `src/include/gasp.hpp`
- Initialization patterns: `src/include/mlop_init.hpp` (constexpr functions)

