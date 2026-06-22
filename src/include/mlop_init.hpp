#pragma once

#include "mlop_config.hpp"
#include "mlop_data_type.hpp"

// ============================================================================
// MLOP Initialization Data Structures and Functions
// ============================================================================
// Constexpr-compatible initialization for MLOP components.
// These structures and functions enable static compile-time initialization
// of arrays that can be reshaped/partitioned by HLS pragmas.
// Follows the same pattern as BOP and SPP for consistency.

// ============================================================================
// Offset Scores Matrix Structure
// ============================================================================
// Stores learned offset scores for each prefetch degree
// Structure: scores[degree][offset]
// Each score tracks how many times an offset predicted a cache hit
//
template<typename mlop_score_t>
struct MLOPOffsetScoresMatrix {
    mlop_score_t scores[MLOP_PF_DEGREE][MLOP_NUM_OFFSETS];
    // Eliminado: MLOPOffsetScoresMatrix(){};
};

// ============================================================================
// Best Offsets Matrix Structure
// ============================================================================
// Stores the best offsets selected after each learning round
// Structure: offsets[degree][index], with counts[degree] tracking how many per degree
//
template<typename mlop_offset_t>
struct MLOPBestOffsetsMatrix {
    mlop_offset_t offsets[MLOP_PF_DEGREE][MLOP_NUM_OFFSETS];
    uint8_t counts[MLOP_PF_DEGREE];
    // Eliminado: MLOPBestOffsetsMatrix(){};
};

// ============================================================================
// Prefetch Level Matrix Structure
// ============================================================================
// Stores the prefetch fill level (L1, L2, LLC) for each degree
//
template<typename level_t>
struct MLOPPrefetchLevelMatrix {
	level_t levels[MLOP_PF_DEGREE];
    // Eliminado: MLOPPrefetchLevelMatrix(){};
};

// ============================================================================
// Update Counter and Round Tracking Matrix
// ============================================================================
// Tracks progress within current learning round
//
template<typename mlop_counter_t>
struct MLOPRoundTrackingMatrix {
    mlop_counter_t update_count;    // Counts accesses in current round
    mlop_counter_t round_count;     // Total completed rounds
    // Eliminado: MLOPRoundTrackingMatrix(){};
};

// ============================================================================
// Access Map Entry Matrix Structure
// ============================================================================
// Stores state of blocks within a zone
// - access_map: State (INIT, ACCESS, PREFETCH) for each block
// - prefetch_map: Fill level for prefetched blocks
//
template<typename mlop_state_t>
struct MLOPAccessMapEntry {
    mlop_state_t access_map[MLOP_BLOCKS_IN_ZONE];
    uint8_t prefetch_map[MLOP_BLOCKS_IN_ZONE];
    // Eliminado: MLOPAccessMapEntry(){};
};

// ============================================================================
// Full Access Map Table Matrix Structure
// ============================================================================
// Stores all zone entries for the access map table
// Each entry tracks block states within its zone
//
template<typename mlop_state_t>
struct MLOPAccessMapTable {
    MLOPAccessMapEntry<mlop_state_t> entries[MLOP_AMT_SIZE];
    uint8_t valid[MLOP_AMT_SIZE];
    uint32_t lru[MLOP_AMT_SIZE];
    // Eliminado: MLOPAccessMapTable(){};
};

// ============================================================================
// Initialization Functions (Constexpr)
// ============================================================================

// ========================================================================
// initMLOPOffsetScores: Create initial offset scores matrix
// ========================================================================
template<typename mlop_score_t>
constexpr MLOPOffsetScoresMatrix<mlop_score_t> initMLOPOffsetScores() {
    MLOPOffsetScoresMatrix<mlop_score_t> res;
    for (int d = 0; d < MLOP_PF_DEGREE; d++) {
        for (int o = 0; o < MLOP_NUM_OFFSETS; o++) {
            res.scores[d][o] = 0;
        }
    }
    return res;
}

// ========================================================================
// initMLOPBestOffsets: Create initial best offsets matrix
// ========================================================================
template<typename mlop_offset_t>
constexpr MLOPBestOffsetsMatrix<mlop_offset_t> initMLOPBestOffsets() {
    MLOPBestOffsetsMatrix<mlop_offset_t> res;
    for (int d = 0; d < MLOP_PF_DEGREE; d++) {
        res.counts[d] = 0;
        for (int o = 0; o < MLOP_NUM_OFFSETS; o++) {
            res.offsets[d][o] = 0;
        }
    }
    return res;
}

// ========================================================================
// initMLOPPrefetchLevels: Create initial prefetch levels matrix
// ========================================================================
template<typename level_t>
inline constexpr MLOPPrefetchLevelMatrix<level_t> initMLOPPrefetchLevels() {
    MLOPPrefetchLevelMatrix<level_t> res;
    for (int d = 0; d < MLOP_PF_DEGREE; d++) {
        res.levels[d] = 0;
    }
    return res;
}

// ========================================================================
// initMLOPRoundTracking: Create initial round tracking
// ========================================================================
template<typename mlop_counter_t>
constexpr MLOPRoundTrackingMatrix<mlop_counter_t> initMLOPRoundTracking() {
    MLOPRoundTrackingMatrix<mlop_counter_t> res;
    res.update_count = 0;
    res.round_count = 0;
    return res;
}

// ========================================================================
// initMLOPAccessMapTable: Create initial access map table
// ========================================================================
template<typename mlop_state_t>
constexpr MLOPAccessMapTable<mlop_state_t> initMLOPAccessMapTable() {
    MLOPAccessMapTable<mlop_state_t> res;
    for (int i = 0; i < MLOP_AMT_SIZE; i++) {
        res.valid[i] = 0;
        res.lru[i] = 0;
        for (int j = 0; j < MLOP_BLOCKS_IN_ZONE; j++) {
            res.entries[i].access_map[j] = MLOP_STATE_INIT;
            res.entries[i].prefetch_map[j] = 0;
        }
    }
    return res;
}

