#pragma once

#include "mlop_config.hpp"
#include "mlop_data_type.hpp"

// ============================================================================
// MLOP Prefetcher - Ultimate Pipelined HLS Implementation
// ============================================================================

template<typename address_t = mlop_address_t,
         typename zone_address_t = mlop_zone_address_t,
         typename zone_offset_t = mlop_zone_offset_t,
         typename offset_t = mlop_offset_t,
         typename score_t = mlop_score_t,
         typename degree_t = mlop_degree_t>
class MLOPrefetcher {
private:
    // ====================================================================
    // Hardware Architecture Geometry
    // ====================================================================
    static constexpr int AMT_WAYS = 16;
    static constexpr int AMT_SETS = (MLOP_AMT_SIZE > AMT_WAYS) ? (MLOP_AMT_SIZE / AMT_WAYS) : 1;
    static constexpr int MAX_OFFSET = MLOP_BLOCKS_IN_ZONE - 1;
    static constexpr int ORIGIN = MAX_OFFSET;

    // ====================================================================
    // Hardware State Variables
    // ====================================================================
    score_t scores[MLOP_PF_DEGREE][MLOP_NUM_OFFSETS];
    offset_t best_offsets[MLOP_PF_DEGREE][MLOP_NUM_OFFSETS];
    uint8_t best_counts[MLOP_PF_DEGREE];
    uint8_t pf_levels[MLOP_PF_DEGREE];

    // Variables nuevas para el "Running Max" (Eliminan bucles de búsqueda)
    score_t max_scores[MLOP_PF_DEGREE];
    offset_t current_best[MLOP_PF_DEGREE];

    zone_address_t amt_tags[AMT_SETS][AMT_WAYS];
    uint8_t amt_lru[AMT_SETS][AMT_WAYS];
    bool amt_valid[AMT_SETS][AMT_WAYS];
    mlop_state_t amt_access_map[AMT_SETS][AMT_WAYS][MLOP_BLOCKS_IN_ZONE];

    zone_offset_t amt_hist[AMT_SETS][AMT_WAYS][MLOP_PF_DEGREE];
    bool amt_hist_valid[AMT_SETS][AMT_WAYS][MLOP_PF_DEGREE];

    mlop_counter_t update_count;

    // ====================================================================
    // Inline Helper Methods
    // ====================================================================

    // Stage 1: AMT Lookup and LRU Management
    uint32_t lookup_amt(uint32_t amt_set, zone_address_t zone_addr, bool &is_new_alloc) {
        #pragma HLS INLINE

        bool hit_vector[AMT_WAYS];
        #pragma HLS ARRAY_PARTITION variable=hit_vector complete

        for (int w = 0; w < AMT_WAYS; w++) {
            #pragma HLS UNROLL
            hit_vector[w] = (amt_valid[amt_set][w] && amt_tags[amt_set][w] == zone_addr);
        }

        uint32_t match_way = 0;
        bool is_hit = false;

        for (int w = 0; w < AMT_WAYS; w++) {
            #pragma HLS UNROLL
            if (hit_vector[w]) {
                match_way |= w;
                is_hit = true;
            }
        }

        is_new_alloc = !is_hit;

        uint8_t tree_val[AMT_WAYS];
        uint8_t tree_idx[AMT_WAYS];
        #pragma HLS ARRAY_PARTITION variable=tree_val complete
        #pragma HLS ARRAY_PARTITION variable=tree_idx complete

        for (int w = 0; w < AMT_WAYS; w++) {
            #pragma HLS UNROLL
            tree_val[w] = amt_lru[amt_set][w];
            tree_idx[w] = w;
        }

        for (int step = 1; step < AMT_WAYS; step *= 2) {
            #pragma HLS UNROLL
            for (int i = 0; i < AMT_WAYS; i += 2 * step) {
                #pragma HLS UNROLL
                if (tree_val[i + step] > tree_val[i]) {
                    tree_val[i] = tree_val[i + step];
                    tree_idx[i] = tree_idx[i + step];
                }
            }
        }

        uint32_t victim_way = tree_idx[0];

        if (is_new_alloc) {
            match_way = victim_way;
            amt_valid[amt_set][match_way] = true;
            amt_tags[amt_set][match_way] = zone_addr;
        }

        for (int w = 0; w < AMT_WAYS; w++) {
            #pragma HLS UNROLL
            if (amt_valid[amt_set][w] && amt_lru[amt_set][w] < 255) {
                amt_lru[amt_set][w]++;
            }
        }
        amt_lru[amt_set][match_way] = 0;

        return match_way;
    }

    // Stage 2 & 3: Learning, Max-Tracking, and Mutually Exclusive Round Evaluation
    void learn_and_evaluate(uint32_t amt_set, uint32_t match_way, zone_offset_t zone_offset, bool is_new_alloc) {
        #pragma HLS INLINE

        mlop_state_t local_map[MLOP_BLOCKS_IN_ZONE];
        #pragma HLS ARRAY_PARTITION variable=local_map complete

        for (int i = 0; i < MLOP_BLOCKS_IN_ZONE; i++) {
            #pragma HLS UNROLL
            if (is_new_alloc) {
                local_map[i] = MLOP_STATE_INIT;
            } else {
                local_map[i] = amt_access_map[amt_set][match_way][i];
            }
        }

        bool is_new_access = (local_map[zone_offset] != MLOP_STATE_ACCESS);

        if (is_new_access) {

            zone_offset_t local_hist[MLOP_PF_DEGREE];
            bool local_hist_valid[MLOP_PF_DEGREE];
            #pragma HLS ARRAY_PARTITION variable=local_hist complete
            #pragma HLS ARRAY_PARTITION variable=local_hist_valid complete

            for (int d = 0; d < MLOP_PF_DEGREE; d++) {
                #pragma HLS UNROLL
                if (is_new_alloc) {
                    local_hist[d] = 0;
                    local_hist_valid[d] = false;
                } else {
                    local_hist[d] = amt_hist[amt_set][match_way][d];
                    local_hist_valid[d] = amt_hist_valid[amt_set][match_way][d];
                }
            }

            // ================================================================
            // Exclusión Mutua: Evita colisiones de lectura y escritura en memoria
            // ================================================================
            if (update_count >= MLOP_NUM_UPDATES) {
                // FASE 1: Final de Ronda (Reseteo sin aprendizaje)
                update_count = 0;

                for (int d = 0; d < MLOP_PF_DEGREE; d++) {
                    #pragma HLS UNROLL
                    if (max_scores[d] >= MLOP_L2C_THRESHOLD) {
                        best_offsets[d][0] = current_best[d];
                        best_counts[d] = 1;
                        pf_levels[d] = 1;
                    } else {
                        best_counts[d] = 0;
                        pf_levels[d] = 0;
                    }

                    max_scores[d] = 0;
                    current_best[d] = 0;

                    for (int i = 0; i < MLOP_NUM_OFFSETS; i++) {
                        #pragma HLS UNROLL
                        scores[d][i] = 0;
                    }
                }
            } else {
                // FASE 2: Aprendizaje estándar (Tracking del "Running Max")
                update_count++;

                for (int d = 0; d < MLOP_PF_DEGREE; d++) {
                    #pragma HLS UNROLL
                    if (local_hist_valid[d]) {
                        int offset_delta = (int)zone_offset - (int)local_hist[d];

                        if (offset_delta != 0 && offset_delta >= -MAX_OFFSET && offset_delta <= MAX_OFFSET) {
                            int idx = ORIGIN + offset_delta;
                            score_t new_score = scores[d][idx] + 1;
                            scores[d][idx] = new_score;

                            // Actualiza el máximo al vuelo, evitando el enorme bucle de búsqueda
                            if (new_score > max_scores[d]) {
                                max_scores[d] = new_score;
                                current_best[d] = offset_delta;
                            }
                        }
                    }
                }
            }

            // El historial se desplaza en todos los accesos independientemente de la exclusión
            for (int d = MLOP_PF_DEGREE - 1; d > 0; d--) {
                #pragma HLS UNROLL
                local_hist[d] = local_hist[d - 1];
                local_hist_valid[d] = local_hist_valid[d - 1];
            }
            local_hist[0] = zone_offset;
            local_hist_valid[0] = true;

            local_map[zone_offset] = MLOP_STATE_ACCESS;

            for (int i = 0; i < MLOP_BLOCKS_IN_ZONE; i++) {
                #pragma HLS UNROLL
                amt_access_map[amt_set][match_way][i] = local_map[i];
            }
            for (int d = 0; d < MLOP_PF_DEGREE; d++) {
                #pragma HLS UNROLL
                amt_hist[amt_set][match_way][d] = local_hist[d];
                amt_hist_valid[amt_set][match_way][d] = local_hist_valid[d];
            }
        }
    }

    // Stage 4: Prefetch Generation
    void generate_prefetches(uint32_t amt_set, uint32_t match_way, zone_offset_t zone_offset,
                                    offset_t* prefetch_deltas, score_t* prefetch_confidences, uint32_t& num_prefetches) {
        #pragma HLS INLINE
        num_prefetches = 0;

        #if MLOP_SINGLE_PREFETCH
            if (best_counts[0] > 0 && pf_levels[0] > 0) {
                offset_t cur_offset = best_offsets[0][0];
                int pf_zone_offset = (int)zone_offset + (int)cur_offset;

                if (pf_zone_offset >= 0 && pf_zone_offset < MLOP_BLOCKS_IN_ZONE && cur_offset != 0) {
                    if (amt_access_map[amt_set][match_way][pf_zone_offset] != MLOP_STATE_ACCESS) {
                        prefetch_deltas[0] = cur_offset;
                        prefetch_confidences[0] = 100;
                        num_prefetches = 1;
                    }
                }
            }
        #else
            uint32_t local_num = 0;
            for (int d = 0; d < MLOP_PF_DEGREE; d++) {
                #pragma HLS UNROLL
                if (pf_levels[d] > 0 && best_counts[d] > 0) {
                    offset_t cur_offset = best_offsets[d][0];
                    int pf_zone_offset = (int)zone_offset + (int)cur_offset;

                    if (pf_zone_offset >= 0 && pf_zone_offset < MLOP_BLOCKS_IN_ZONE && cur_offset != 0) {
                        if (amt_access_map[amt_set][match_way][pf_zone_offset] != MLOP_STATE_ACCESS) {
                            prefetch_deltas[d] = cur_offset;
                            prefetch_confidences[d] = 100;
                            local_num++;
                        }
                    }
                }
            }
            num_prefetches = local_num;
        #endif
    }

public:
    MLOPrefetcher() {
        update_count = 0;
        for(int d=0; d<MLOP_PF_DEGREE; d++){
            max_scores[d] = 0;
            current_best[d] = 0;
        }
    }

    void process_cache_access(address_t addr,
                             offset_t* prefetch_deltas,
                             score_t* prefetch_confidences,
                             uint32_t& num_prefetches) {
        #pragma HLS PIPELINE II=1

        // ====================================================================
        // Partitioning Pragmas (dim=0 asegurado sin comentarios que lo rompan)
        // ====================================================================
        #pragma HLS ARRAY_PARTITION variable=scores complete dim=0
        #pragma HLS ARRAY_PARTITION variable=best_offsets complete dim=0
        #pragma HLS ARRAY_PARTITION variable=best_counts complete dim=0
        #pragma HLS ARRAY_PARTITION variable=pf_levels complete dim=0

        #pragma HLS ARRAY_PARTITION variable=max_scores complete dim=0
        #pragma HLS ARRAY_PARTITION variable=current_best complete dim=0

        #pragma HLS ARRAY_PARTITION variable=amt_tags complete dim=2
        #pragma HLS ARRAY_PARTITION variable=amt_lru complete dim=2
        #pragma HLS ARRAY_PARTITION variable=amt_valid complete dim=2

        #pragma HLS ARRAY_PARTITION variable=amt_access_map complete dim=2
        #pragma HLS ARRAY_PARTITION variable=amt_access_map complete dim=3

        #pragma HLS ARRAY_PARTITION variable=amt_hist complete dim=2
        #pragma HLS ARRAY_PARTITION variable=amt_hist complete dim=3

        #pragma HLS ARRAY_PARTITION variable=amt_hist_valid complete dim=2
        #pragma HLS ARRAY_PARTITION variable=amt_hist_valid complete dim=3

        // ====================================================================
        // Execution
        // ====================================================================
        zone_address_t zone_addr = addr >> MLOP_LOG2_PAGE_SIZE;
        zone_offset_t zone_offset = (addr >> MLOP_LOG2_BLOCK_SIZE) & (MLOP_BLOCKS_IN_ZONE - 1);
        uint32_t amt_set = zone_addr % AMT_SETS;

        bool is_new_alloc = false;
        uint32_t match_way = lookup_amt(amt_set, zone_addr, is_new_alloc);

        // Aprende, lleva récord de los máximos y se auto-resetea mutuamente
        learn_and_evaluate(amt_set, match_way, zone_offset, is_new_alloc);

        generate_prefetches(amt_set, match_way, zone_offset, prefetch_deltas, prefetch_confidences, num_prefetches);
    }

    void register_fill(address_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr) {
    }
};
