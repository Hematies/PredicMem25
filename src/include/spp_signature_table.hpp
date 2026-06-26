#pragma once

#include "spp_config.hpp"
#include "spp_data_type.hpp"

// ============================================================================
// SPP Signature Table (ST)
// ============================================================================
// Stores per-page information including:
// - Page tag (partial page address)
// - Signature for pattern correlation
// - Last cache block offset within the page
// - LRU replacement information

// Forward declaration of matrix struct (defined in spp_init.hpp)
// struct SPPSignatureTableMatrix;

template<typename st_tag_t = spp_st_tag_t, typename st_sig_t = spp_st_sig_t, typename st_confidence_t = spp_st_confidence_t>
class SPPSignatureTable {
public:
	spp_ghr_valid_t valid[SPP_ST_SET][SPP_ST_WAY];
	    //// #pragma HLS ARRAY_PARTITION variable=valid complete dim=2

	    st_tag_t tag[SPP_ST_SET][SPP_ST_WAY];
	    //// #pragma HLS ARRAY_PARTITION variable=tag complete dim=2

	    spp_page_offset_t last_offset[SPP_ST_SET][SPP_ST_WAY];
	    //// #pragma HLS ARRAY_PARTITION variable=last_offset complete dim=2

	    st_sig_t sig[SPP_ST_SET][SPP_ST_WAY];
	    //// #pragma HLS ARRAY_PARTITION variable=sig complete dim=2

	    spp_st_lru_t lru[SPP_ST_SET][SPP_ST_WAY];
	    // // #pragma HLS ARRAY_PARTITION variable=lru complete dim=2

    // Default constructor - initialization via constexpr in caller
    SPPSignatureTable() = default;

    static uint64_t hash_address(uint64_t key) {
		#pragma HLS PIPELINE
		// Las primeras líneas de tu hash actual...
		key += (key << 12);
		key ^= (key >> 22);
		key += (key << 4);
		key ^= (key >> 9);
		key += (key << 10);
		key ^= (key >> 2);
		key += (key << 7);
		key ^= (key >> 12);

		// --- SUSTITUCIÓN DE LA MULTIPLICACIÓN ---
		// key = (key >> 3) * 2654435761ULL;
		// La constante 2654435761 (aprox 0x9E3779B1) se puede descomponer:
		key = (key >> 3);
		key = (key << 16) - key + (key << 8) + (key << 4) + (key << 1);

		return key;
	}

    // Read and update signature based on new page access
	// Returns: last_sig (previous signature), curr_sig (new signature), delta (offset difference)
	void read_and_update_sig(spp_address_t page, spp_page_offset_t page_offset,
							 st_sig_t& last_sig, st_sig_t& curr_sig,
							 spp_pt_delta_t& delta) {
		#pragma HLS INLINE

		spp_st_set_index_t set = hash_address(page) % SPP_ST_SET;
		st_tag_t partial_page = page & SPP_ST_TAG_MASK;

		uint32_t match_way = SPP_ST_WAY;
		uint32_t invalid_way = SPP_ST_WAY;
		uint32_t victim_way = SPP_ST_WAY;

		// ====================================================================
		// Stage 1 & 3: Extracción Paralela Limpia
		// ====================================================================
		bool hit_vector[SPP_ST_WAY];
		bool inv_vector[SPP_ST_WAY];
		#pragma HLS ARRAY_PARTITION variable=hit_vector complete
		#pragma HLS ARRAY_PARTITION variable=inv_vector complete

		// Leemos todo el estado de la memoria en paralelo (sin dependencias cruzadas)
		for (uint32_t way = 0; way < SPP_ST_WAY; way++) {
			#pragma HLS UNROLL
			hit_vector[way] = (valid[set][way] && (tag[set][way] == partial_page));
			inv_vector[way] = !valid[set][way];

			if (lru[set][way] == (SPP_ST_WAY - 1)) {
				victim_way = way;
			}
		}

		// Priority Encoder: Bucle en reversa para encontrar el PRIMER hit o inválido.
		// HLS lo sintetiza como un árbol lógico súper rápido, no como un bucle.
		for (int way = SPP_ST_WAY - 1; way >= 0; way--) {
			#pragma HLS UNROLL
			if (hit_vector[way]) match_way = way;
			if (inv_vector[way]) invalid_way = way;
		}

		// ====================================================================
		// Stage 2: Hit / Miss Logic
		// ====================================================================
		if (match_way < SPP_ST_WAY) {
			// --- HIT ---
			last_sig = sig[set][match_way];
			delta = page_offset - last_offset[set][match_way];

			if (delta != 0) {
				// Generate signature delta with 7-bit sign magnitude representation
				spp_sig_delta_t sig_delta = (delta < 0) ?
					(spp_sig_delta_t)(((-delta) & 0x3F) | 0x40) : delta;

				sig[set][match_way] = ((last_sig << SPP_SIG_SHIFT) ^ sig_delta) & SPP_SIG_MASK;
			}

			curr_sig = sig[set][match_way];
			last_offset[set][match_way] = page_offset;

		} else {
			// --- MISS ---
			// Resolvemos el reemplazo con lógica ternaria simple
			match_way = (invalid_way < SPP_ST_WAY) ? invalid_way : victim_way;

			if (match_way < SPP_ST_WAY) {
				valid[set][match_way] = 1;
				tag[set][match_way] = partial_page;
				sig[set][match_way] = 0;
				last_offset[set][match_way] = page_offset;
				curr_sig = 0;
				last_sig = 0;
				delta = 0;
			}
		}

		// ====================================================================
		// Stage 4: Update LRU
		// ====================================================================
		if (match_way < SPP_ST_WAY) {

			// EL TRUCO VITAL DE HARDWARE: Leer la memoria objetivo en una variable
			// escalar ANTES del bucle desenrollado. Esto rompe la matriz de multiplexores.
			spp_st_lru_t target_lru = lru[set][match_way];

			for (uint32_t way = 0; way < SPP_ST_WAY; way++) {
				#pragma HLS UNROLL
				if (valid[set][way] && lru[set][way] < target_lru) {
					lru[set][way]++;
				}
			}
			lru[set][match_way] = 0;  // Promote to MRU
		}
	}
};
