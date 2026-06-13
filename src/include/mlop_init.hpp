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
struct MLOPOffsetScoresMatrix {
    mlop_score_t scores[MLOP_PF_DEGREE][MLOP_NUM_OFFSETS];

    // Constructor for constexpr initialization
    constexpr MLOPOffsetScoresMatrix() {
        for (int d = 0; d < MLOP_PF_DEGREE; d++) {
            for (int o = 0; o < MLOP_NUM_OFFSETS; o++) {
                scores[d][o] = 0;
            }
        }
    }
};

// ============================================================================
// Best Offsets Matrix Structure
// ============================================================================
// Stores the best offsets selected after each learning round
// Structure: offsets[degree][index], with counts[degree] tracking how many per degree
//
struct MLOPBestOffsetsMatrix {
    mlop_offset_t offsets[MLOP_PF_DEGREE][MLOP_NUM_OFFSETS];
    uint8_t counts[MLOP_PF_DEGREE];

    // Constructor for constexpr initialization
    constexpr MLOPBestOffsetsMatrix() {
        for (int d = 0; d < MLOP_PF_DEGREE; d++) {
            counts[d] = 0;
            for (int o = 0; o < MLOP_NUM_OFFSETS; o++) {
                offsets[d][o] = 0;
            }
        }
    }
};

// ============================================================================
// Prefetch Level Matrix Structure
// ============================================================================
// Stores the prefetch fill level (L1, L2, LLC) for each degree
//
struct MLOPPrefetchLevelMatrix {
    uint8_t levels[MLOP_PF_DEGREE];

    // Constructor for constexpr initialization
    constexpr MLOPPrefetchLevelMatrix() {
        for (int d = 0; d < MLOP_PF_DEGREE; d++) {
            levels[d] = 0;
        }
    }
};

// ============================================================================
// Update Counter and Round Tracking Matrix
// ============================================================================
// Tracks progress within current learning round
//
struct MLOPRoundTrackingMatrix {
    mlop_counter_t update_count;    // Counts accesses in current round
    mlop_counter_t round_count;     // Total completed rounds

    // Constructor for constexpr initialization
    constexpr MLOPRoundTrackingMatrix() : update_count(0), round_count(0) {
    }
};

// ============================================================================
// Access Map Entry Matrix Structure
// ============================================================================
// Stores state of blocks within a zone
// - access_map: State (INIT, ACCESS, PREFETCH) for each block
// - prefetch_map: Fill level for prefetched blocks
//
struct MLOPAccessMapEntry {
    mlop_state_t access_map[MLOP_BLOCKS_IN_ZONE];
    uint8_t prefetch_map[MLOP_BLOCKS_IN_ZONE];

    // Constructor for constexpr initialization
    constexpr MLOPAccessMapEntry() {
        for (int i = 0; i < MLOP_BLOCKS_IN_ZONE; i++) {
            access_map[i] = MLOP_STATE_INIT;
            prefetch_map[i] = 0;
        }
    }
};

// ============================================================================
// Full Access Map Table Matrix Structure
// ============================================================================
// Stores all zone entries for the access map table
// Each entry tracks block states within its zone
//
struct MLOPAccessMapTable {
    MLOPAccessMapEntry entries[MLOP_AMT_SIZE];
    uint8_t valid[MLOP_AMT_SIZE];
    uint32_t lru[MLOP_AMT_SIZE];

    // Constructor for constexpr initialization
    constexpr MLOPAccessMapTable() {
        for (int i = 0; i < MLOP_AMT_SIZE; i++) {
            valid[i] = 0;
            lru[i] = 0;
            entries[i] = MLOPAccessMapEntry();
        }
    }
};

// ============================================================================
// Initialization Functions (Constexpr)
// ============================================================================

// ========================================================================
// initMLOPOffsetScores: Create initial offset scores matrix
// ========================================================================
inline constexpr MLOPOffsetScoresMatrix initMLOPOffsetScores() {
    return MLOPOffsetScoresMatrix();
}

// ========================================================================
// initMLOPBestOffsets: Create initial best offsets matrix
// ========================================================================
inline constexpr MLOPBestOffsetsMatrix initMLOPBestOffsets() {
    return MLOPBestOffsetsMatrix();
}

// ========================================================================
// initMLOPPrefetchLevels: Create initial prefetch levels matrix
// ========================================================================
inline constexpr MLOPPrefetchLevelMatrix initMLOPPrefetchLevels() {
    return MLOPPrefetchLevelMatrix();
}

// ========================================================================
// initMLOPRoundTracking: Create initial round tracking
// ========================================================================
inline constexpr MLOPRoundTrackingMatrix initMLOPRoundTracking() {
    return MLOPRoundTrackingMatrix();
}

// ========================================================================
// initMLOPAccessMapTable: Create initial access map table
// ========================================================================
inline constexpr MLOPAccessMapTable initMLOPAccessMapTable() {
    return MLOPAccessMapTable();
}

// ============================================================================
// Threshold Pre-computation (Constexpr Static)
// ============================================================================

// Pre-computed thresholds (computed at compile-time)
constexpr mlop_threshold_t MLOP_L1D_THRESHOLD = (mlop_threshold_t)(MLOP_L1D_THRESH * MLOP_NUM_UPDATES);
constexpr mlop_threshold_t MLOP_L2C_THRESHOLD = (mlop_threshold_t)(MLOP_L2C_THRESH * MLOP_NUM_UPDATES);
constexpr mlop_threshold_t MLOP_LLC_THRESHOLD = (mlop_threshold_t)(MLOP_LLC_THRESH * MLOP_NUM_UPDATES);



