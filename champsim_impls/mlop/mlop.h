/* Based on Multi-Lookahead Offset Prefetcher (MLOP) - 3rd Data Prefetching Championship */
/* Owners: Mehran Shakerinava and Mohammad Bakhshalipour */


#ifndef MLOP_H
#define MLOP_H

#include <vector>
#include <unordered_map>
#include <sstream>

#include <iostream>
#include "champsim.h"

#include <algorithm>
#include "cache.h"
#include "bakshalipour_framework.h"
#include<math.h>

/**
 * The access map table records blocks as being in one of 3 general states:
 * ACCESS, PREFETCH, or INIT.
 * The PREFETCH state is actually composed of up to 3 sub-states:
 * L1-PREFETCH, L2-PREFETCH, or L3-PREFETCH.
 * This version of MLOP does not prefetch into L3 so there are 4 states in total (2-bit states).
 */
enum MLOP_State { INIT = 0, ACCESS = 1, PREFTCH = 2 };
char getStateChar(MLOP_State state);
string map_to_string(const vector<MLOP_State> &access_map, const vector<int> &prefetch_map);

class AccessMapData {
  public:
    /* block states are represented with a `MLOP_State` and an `int` in this software implementation but
     * in a hardware implementation, they'd be represented with only 2 bits. */
    vector<MLOP_State> access_map;
    vector<int> prefetch_map;

    deque<int> hist_queue;
};

class AccessMapTable : public LRUSetAssociativeCache<AccessMapData> {
    typedef LRUSetAssociativeCache<AccessMapData> Super;

  public:
    /* NOTE: zones are equivalent to pages (64 blocks) in this implementation */
    AccessMapTable(int size, int blocks_in_zone, int queue_size, int debug_level = 0, int num_ways = 16)
        : Super(size, num_ways, debug_level), blocks_in_zone(blocks_in_zone), queue_size(queue_size) {
        if (this->debug_level >= 1)
            cout << "AccessMapTable::AccessMapTable(size=" << size << ", blocks_in_zone=" << blocks_in_zone
                 << ", queue_size=" << queue_size << ", debug_level=" << debug_level << ", num_ways=" << num_ways << ")"
                 << endl;
    }

    /**
     * Sets specified block to given state. If new state is ACCESS, the block will also be pushed in the zone's queue.
     */
    void set_state(uint64_t block_number, MLOP_State new_state, int new_fill_level = 0) {
        if (this->debug_level >= 2)
            cout << "AccessMapTable::set_state(block_number=0x" << hex << block_number
                 << ", new_state=" << getStateChar(new_state) << ", new_fill_level=" << new_fill_level << ")" << dec
                 << endl;

        // if (new_state != MLOP_State::PREFTCH)
        //     assert(new_fill_level == 0);
        // else
        //     assert(new_fill_level == FILL_L1 || new_fill_level == FILL_L2 || new_fill_level == FILL_LLC);

        uint64_t zone_number = block_number / this->blocks_in_zone;
        int zone_offset = block_number % this->blocks_in_zone;

        uint64_t key = this->build_key(zone_number);
        Entry *entry = Super::find(key);
        if (!entry) {
            // assert(new_state != MLOP_State::PREFTCH);
            if (new_state == MLOP_State::INIT)
                return;
            Super::insert(key, {vector<MLOP_State>(blocks_in_zone, MLOP_State::INIT), vector<int>(blocks_in_zone, 0)});
            entry = Super::find(key);
            // assert(entry->data.hist_queue.empty());
        }

        auto &access_map = entry->data.access_map;
        auto &prefetch_map = entry->data.prefetch_map;
        auto &hist_queue = entry->data.hist_queue;

        if (new_state == MLOP_State::ACCESS) {
            Super::set_mru(key);

            /* insert access into queue */
            hist_queue.push_front(zone_offset);
            if (hist_queue.size() > this->queue_size)
                hist_queue.pop_back();
        }

        MLOP_State old_state = access_map[zone_offset];
        int old_fill_level = prefetch_map[zone_offset];

        vector<MLOP_State> old_access_map = access_map;
        vector<int> old_prefetch_map = prefetch_map;

        access_map[zone_offset] = new_state;
        prefetch_map[zone_offset] = new_fill_level;

        if (new_state == MLOP_State::INIT) {
            /* delete entry if access map is empty (all in state INIT) */
            bool all_init = true;
            for (unsigned i = 0; i < this->blocks_in_zone; i += 1)
                if (access_map[i] != MLOP_State::INIT) {
                    all_init = false;
                    break;
                }
            if (all_init)
                Super::erase(key);
        }

        if (this->debug_level >= 2) {
            cout << "[AccessMapTable::set_state] zone_number=0x" << hex << zone_number << dec
                 << ", zone_offset=" << setw(2) << zone_offset << ": state transition from " << getStateChar(old_state)
                 << " to " << getStateChar(new_state) << endl;
            if (old_state != new_state || old_fill_level != new_fill_level) {
                cout << "[AccessMapTable::set_state] old_access_map=" << map_to_string(old_access_map, old_prefetch_map)
                     << endl;
                cout << "[AccessMapTable::set_state] new_access_map=" << map_to_string(access_map, prefetch_map)
                     << endl;
            }
        }
    }

    Entry *find(uint64_t zone_number) {
        if (this->debug_level >= 2)
            cout << "AccessMapTable::find(zone_number=0x" << hex << zone_number << ")" << dec << endl;
        uint64_t key = this->build_key(zone_number);
        return Super::find(key);
    }

    string log() {
        vector<string> headers({"Zone", "Access Map"});
        return Super::log(headers);
    }

  private:
    /* @override */
    void write_data(Entry &entry, Table &table, int row) {
        uint64_t zone_number = hash_index(entry.key, this->index_len);
        table.set_cell(row, 0, zone_number);
        table.set_cell(row, 1, map_to_string(entry.data.access_map, entry.data.prefetch_map));
    }

    uint64_t build_key(uint64_t zone_number) {
        uint64_t key = zone_number; /* no truncation (52 bits) */
        return hash_index(key, this->index_len);
    }

    unsigned blocks_in_zone;
    unsigned queue_size;

    /*===================================================================*/
    /* Entry   = [tag, map, queue, valid, LRU]                           */
    /* Storage = size * (52 - lg(sets) + 64 * 2 + 15 * 6 + 1 + lg(ways)) */
    /* L1D: 256 * (52 - lg(16) + 128 + 90 + 1 + lg(16)) = 8672 Bytes     */
    /*===================================================================*/
};

class MLOP
{
private:
    inline bool is_inside_zone(uint32_t zone_offset) { return (0 <= zone_offset && zone_offset < this->blocks_in_zone); }

	uint32_t PF_DEGREE;
	uint32_t NUM_UPDATES;
	uint32_t L1D_THRESH;
	uint32_t L2C_THRESH;
	uint32_t LLC_THRESH;
	uint32_t ORIGIN;
	uint32_t NUM_OFFSETS;
    CACHE *parent;

	int32_t MAX_OFFSET, MIN_OFFSET;
    uint32_t blocks_in_cache, blocks_in_zone, amt_size;
    AccessMapTable *access_map_table;

    /**
     * Contains best offsets for each degree of prefetching. A degree will have several offsets if
     * they all had maximum score (thus, a vector of vectors). A degree won't have any offsets if
     * all best offsets were redundant (already selected in previous degrees).
     */
    vector<vector<int>> pf_offset;

    vector<vector<int>> offset_scores;
    vector<int> pf_level; /* the prefetching level for each degree of prefetching offsets */
    uint32_t update_cnt = 0;  /* tracks the number of score updates, round is over when `update_cnt == NUM_UPDATES` */

    uint32_t debug_level = 0;

    /* stats */
    const uint64_t TRACKED_ZONE_CNT = 100;
    bool tracking = false;
    uint64_t tracked_zone_number = 0;
    uint64_t zone_cnt = 0;
    vector<string> zone_life;

    uint64_t round_cnt = 0;
    uint64_t pf_degree_sum = 0, pf_degree_sqr_sum = 0;
    uint64_t max_score_le_sum = 0, max_score_le_sqr_sum = 0;
    uint64_t max_score_ri_sum = 0, max_score_ri_sqr_sum = 0;

private:
	void init_knobs();
	void init_stats();

public:
	MLOP(CACHE *cache);
	~MLOP();
	void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, std::vector<uint64_t> &pref_addr);
    void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, std::vector<int64_t> &offsets);
	void register_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr);
	void dump_stats();
	void print_config();

	void access(uint64_t block_number);
	vector<uint64_t> prefetch(CACHE *cache, uint64_t block_number);
    vector<int64_t> prefetch_offsets(CACHE *cache, uint64_t block_number);
    void mark(uint64_t block_number, MLOP_State state, int fill_level = 0);
    void set_debug_level(int debug_level);
    string log_offset_scores();
    void log();
    void track(uint64_t block_number);
    void reset_stats();
    void print_stats();
};

namespace knob
{
	extern uint32_t mlop_pref_degree;
	extern uint32_t mlop_num_updates;
	extern float 	mlop_l1d_thresh;
	extern float 	mlop_l2c_thresh;
	extern float 	mlop_llc_thresh;
	extern uint32_t	mlop_debug_level;
}

char state_char[] = {'I', 'A', 'P'};
char getStateChar(MLOP_State state) {return state_char[(int)state];}

string map_to_string(const vector<MLOP_State> &access_map, const vector<int> &prefetch_map) {
    ostringstream oss;
    for (unsigned i = 0; i < access_map.size(); i += 1)
        if (access_map[i] == MLOP_State::PREFTCH) {
            oss << prefetch_map[i];
        } else {
            oss << state_char[access_map[i]];
        }
    return oss.str();
}

void MLOP::init_knobs()
{
	PF_DEGREE = knob::mlop_pref_degree;
	NUM_UPDATES = knob::mlop_num_updates;
	L1D_THRESH = knob::mlop_l1d_thresh * NUM_UPDATES;
	L2C_THRESH = knob::mlop_l2c_thresh * NUM_UPDATES;
	LLC_THRESH = knob::mlop_llc_thresh * NUM_UPDATES;
	debug_level = knob::mlop_debug_level;

	blocks_in_cache = parent->NUM_SET * parent->NUM_WAY;
	blocks_in_zone = PAGE_SIZE/BLOCK_SIZE;
	amt_size = 32 * blocks_in_cache / blocks_in_zone;
	ORIGIN = blocks_in_zone - 1;
	MAX_OFFSET = blocks_in_zone - 1;
	MIN_OFFSET = (-1) * MAX_OFFSET;
	NUM_OFFSETS = 2*blocks_in_zone - 1;
}

void MLOP::init_stats()
{

}

MLOP::MLOP(CACHE *cache) : parent(cache)
{
	init_knobs();
	init_stats();

	/* init data structures */
	access_map_table = new AccessMapTable(amt_size, blocks_in_zone, PF_DEGREE - 1, debug_level);
	pf_offset = vector<vector<int>>(PF_DEGREE, vector<int>());
	pf_level = vector<int>(PF_DEGREE, 0);
	offset_scores = vector<vector<int>>(PF_DEGREE, vector<int>(NUM_OFFSETS, 0));
}

MLOP::~MLOP()
{

}

void MLOP::print_config()
{
	cout << "mlop_pref_degree " << knob::mlop_pref_degree << endl
		<< "mlop_num_updates " << knob::mlop_num_updates << endl
		<< "mlop_l1d_thresh " << knob::mlop_l1d_thresh << endl
		<< "mlop_l2c_thresh " << knob::mlop_l2c_thresh << endl
		<< "mlop_llc_thresh " << knob::mlop_llc_thresh << endl
		<< "mlop_debug_level " << knob::mlop_debug_level << endl
		<< "mlop_blocks_in_cache " << blocks_in_cache << endl
		<< "mlop_blocks_in_zone " << blocks_in_zone << endl
		<< "mlop_amt_size " << amt_size << endl
		<< "mlop_PF_DEGREE " << PF_DEGREE << endl
		<< "mlop_NUM_UPDATES " << NUM_UPDATES << endl
		<< "mlop_L1D_THRESH " << L1D_THRESH << endl
		<< "mlop_L2C_THRESH " << L2C_THRESH << endl
		<< "mlop_LLC_THRESH " << LLC_THRESH << endl
		<< "mlop_ORIGIN " << ORIGIN << endl
		<< "mlop_MAX_OFFSET " << MAX_OFFSET << endl
		<< "mlop_MIN_OFFSET " << MIN_OFFSET << endl
		<< "mlop_ORIGIN " << ORIGIN << endl
		<< "mlop_NUM_OFFSETS " << NUM_OFFSETS << endl
		<< endl;
}

/**
 * Updates MLOP's state based on the most recent trigger access (LOAD miss/prefetch-hit).
 * @param block_number The block address of the most recent trigger access
 */
void MLOP::access(uint64_t block_number) {
	if (this->debug_level >= 2)
		cout << "MLOP::access(block_number=0x" << hex << block_number << ")" << dec << endl;

	uint64_t zone_number = block_number / this->blocks_in_zone;
	int zone_offset = block_number % this->blocks_in_zone;

	if (this->debug_level >= 2)
		cout << "[MLOP::access] zone_number=0x" << hex << zone_number << dec << ", zone_offset=" << zone_offset
			<< endl;

	// for (int i = 0; i < PF_DEGREE; i += 1)
	//     assert(this->offset_scores[i][ORIGIN] == 0);

	/* update scores */
	AccessMapTable::Entry *entry = this->access_map_table->find(zone_number);
	if (!entry) {
		/* stats */
		this->zone_cnt += 1;
		if (this->zone_cnt == TRACKED_ZONE_CNT) {
			this->tracked_zone_number = zone_number;
			this->tracking = true;
			this->zone_life.push_back(string(this->blocks_in_zone, state_char[MLOP_State::INIT]));
		}
		/* ===== */
		return;
	}
	vector<MLOP_State> access_map = entry->data.access_map;
	if (access_map[zone_offset] == MLOP_State::ACCESS)
		return; /* ignore repeated trigger access */
	this->update_cnt += 1;
	const deque<int> &queue = entry->data.hist_queue;
	for (int d = 0; d <= (int)queue.size(); d += 1) {
		/* unmark latest access to increase prediction depth */
		if (d != 0) {
			int idx = queue[d - 1];
			// assert(0 <= idx && idx < this->blocks_in_zone);
			access_map[idx] = MLOP_State::INIT;
		}
		// cout << "inside access" << endl;
		for (uint32_t i = 0; i < this->blocks_in_zone; i += 1) {
			if (access_map[i] == MLOP_State::ACCESS) {
				int offset = zone_offset - i;
				// cout << "offset " << offset << endl;
				if (offset >= MIN_OFFSET && offset <= MAX_OFFSET && offset != 0)
				{
					// cout << "incrementing offset_score" << endl;
					this->offset_scores[d][ORIGIN + offset] += 1;
				}
			}
		}
	}

	/* update prefetching offsets if round is finished */
	if (this->update_cnt == NUM_UPDATES) {
		if (this->debug_level >= 1)
			cout << "[MLOP::access] Round finished!" << endl;

		/* reset `update_cnt` and clear `pf_level` and `pf_offset` */
		this->update_cnt = 0;
		this->pf_level = vector<int>(PF_DEGREE, 0);
		this->pf_offset = vector<vector<int>>(PF_DEGREE, vector<int>());

		/* calculate maximum score for all degrees */
		vector<int> max_scores(PF_DEGREE, 0);
		for (uint32_t i = 0; i < PF_DEGREE; i += 1) {
			max_scores[i] = *max_element(this->offset_scores[i].begin(), this->offset_scores[i].end());
			/* `max_scores` should be decreasing */
			// if (i > 0)
			//     assert(max_scores[i] <= max_scores[i - 1]);
		}

		int fill_level = 0;
		vector<bool> pf_offset_map(NUM_OFFSETS, false);
		for (int d = PF_DEGREE - 1; d >= 0; d -= 1) {
			// cout << "d " << d << " max_scores[d] " << max_scores[d] << endl;
			/* check thresholds */
			// if (max_scores[d] >= (int)L1D_THRESH)
			// 	fill_level = FILL_L1;
			if (max_scores[d] >= (int)L2C_THRESH)
				fill_level = 0;
			else if (max_scores[d] >= (int)LLC_THRESH)
				fill_level = 0;
			else
				continue;

			/* select offsets with highest score */
			vector<int> best_offsets;
			for (int i = MIN_OFFSET; i <= MAX_OFFSET; i += 1) {
				int &cur_score = this->offset_scores[d][ORIGIN + i];
				// assert(0 <= cur_score && cur_score <= NUM_UPDATES);
				if (cur_score == max_scores[d] && !pf_offset_map[ORIGIN + i])
					best_offsets.push_back(i);
			}

			// cout << "came here too!" << endl;
			this->pf_level[d] = fill_level;
			this->pf_offset[d] = best_offsets;

			/* mark in `pf_offset_map` to avoid duplicate prefetch offsets */
			for (int i = 0; i < (int)best_offsets.size(); i += 1)
				pf_offset_map[ORIGIN + best_offsets[i]] = true;
		}

		/* reset `offset_scores` */
		this->offset_scores = vector<vector<int>>(PF_DEGREE, vector<int>(NUM_OFFSETS, 0));

		/* print selected prefetching offsets if debug is on */
		if (this->debug_level >= 1) {
			for (uint32_t d = 0; d < PF_DEGREE; d += 1) {
				cout << "[MLOP::access] Degree=" << setw(2) << d + 1;
				cout << ", Offsets: ";
				if (this->pf_offset[d].size() == 0)
					cout << "None";
				for (int i = 0; i < (int)this->pf_offset[d].size(); i += 1) {
					cout << this->pf_offset[d][i];
					if (i < (int)this->pf_offset[d].size() - 1)
						cout << ", ";
				}
				cout << endl;
			}
		}

		/* stats */
		this->round_cnt += 1;

		int cur_pf_degree = 0;
		for (const bool &x : pf_offset_map)
			cur_pf_degree += (x ? 1 : 0);
		this->pf_degree_sum += cur_pf_degree;
		this->pf_degree_sqr_sum += square(cur_pf_degree);

		uint64_t max_score_le = max_scores[PF_DEGREE - 1];
		uint64_t max_score_ri = max_scores[0];
		this->max_score_le_sum += max_score_le;
		this->max_score_ri_sum += max_score_ri;
		this->max_score_le_sqr_sum += square(max_score_le);
		this->max_score_ri_sqr_sum += square(max_score_ri);
		/* ===== */
	}
}

/**
 * @param block_number The block address of the most recent LOAD access
 */
vector<int64_t> MLOP::prefetch_offsets(CACHE *cache, uint64_t block_number) {
    vector<int64_t> res = vector<int64_t>();
	if (this->debug_level >= 2) {
		cout << "MLOP::prefetch(cache=" << cache->NAME << "-" << cache->cpu << ", block_number=0x" << hex
			<< block_number << dec << ")" << endl;
	}
	int pf_issued = 0;
	uint64_t zone_number = block_number / this->blocks_in_zone;
	int zone_offset = block_number % this->blocks_in_zone;
	AccessMapTable::Entry *entry = this->access_map_table->find(zone_number);
	// assert(entry); /* I expect `mark` to have been called before `prefetch` */
	const vector<MLOP_State> &access_map = entry->data.access_map;
	const vector<int> &prefetch_map = entry->data.prefetch_map;
	if (this->debug_level >= 2) {
		cout << "[MLOP::prefetch] old_access_map=" << map_to_string(access_map, prefetch_map) << endl;
	}
	for (uint32_t d = 0; d < PF_DEGREE; d += 1) {
		for (auto &cur_pf_offset : this->pf_offset[d]) {
			// assert(this->pf_level[d] > 0);
			int offset_to_prefetch = zone_offset + cur_pf_offset;

			/* use `access_map` to filter prefetches */
			if (access_map[offset_to_prefetch] == MLOP_State::ACCESS)
				continue;
			if (access_map[offset_to_prefetch] == MLOP_State::PREFTCH &&
					prefetch_map[offset_to_prefetch] <= this->pf_level[d])
				continue;

			if (this->is_inside_zone(offset_to_prefetch) && cache->get_pq_occupancy()[0] < cache->PQ_SIZE &&
					cache->get_pq_occupancy()[0] + cache->get_mshr_occupancy() < cache->MSHR_SIZE - 1) {
				uint64_t pf_block_number = block_number + cur_pf_offset;
				uint64_t base_addr = block_number << LOG2_BLOCK_SIZE;
				uint64_t pf_addr = pf_block_number << LOG2_BLOCK_SIZE;
                res.push_back(cur_pf_offset << LOG2_BLOCK_SIZE);
				// cache->prefetch_line(0, base_addr, pf_addr, this->pf_level[d], 0);
				// assert(ok == 1);
				this->mark(pf_block_number, MLOP_State::PREFTCH, this->pf_level[d]);
				pf_issued += 1;
			}
		}
	}
	if (this->debug_level >= 2) {
		cout << "[MLOP::prefetch] new_access_map=" << map_to_string(access_map, prefetch_map) << endl;
		cout << "[MLOP::prefetch] issued " << pf_issued << " prefetch(es)" << endl;
	}
    return res;
}

/**
 * @param block_number The block address of the most recent LOAD access
 */
vector<uint64_t> MLOP::prefetch(CACHE *cache, uint64_t block_number) {
    vector<uint64_t> res = vector<uint64_t>();
	if (this->debug_level >= 2) {
		cout << "MLOP::prefetch(cache=" << cache->NAME << "-" << cache->cpu << ", block_number=0x" << hex
			<< block_number << dec << ")" << endl;
	}
	int pf_issued = 0;
	uint64_t zone_number = block_number / this->blocks_in_zone;
	int zone_offset = block_number % this->blocks_in_zone;
	AccessMapTable::Entry *entry = this->access_map_table->find(zone_number);
	// assert(entry); /* I expect `mark` to have been called before `prefetch` */
	const vector<MLOP_State> &access_map = entry->data.access_map;
	const vector<int> &prefetch_map = entry->data.prefetch_map;
	if (this->debug_level >= 2) {
		cout << "[MLOP::prefetch] old_access_map=" << map_to_string(access_map, prefetch_map) << endl;
	}
	for (uint32_t d = 0; d < PF_DEGREE; d += 1) {
		for (auto &cur_pf_offset : this->pf_offset[d]) {
			// assert(this->pf_level[d] > 0);
			int offset_to_prefetch = zone_offset + cur_pf_offset;

			/* use `access_map` to filter prefetches */
			if (access_map[offset_to_prefetch] == MLOP_State::ACCESS)
				continue;
			if (access_map[offset_to_prefetch] == MLOP_State::PREFTCH &&
					prefetch_map[offset_to_prefetch] <= this->pf_level[d])
				continue;

			if (this->is_inside_zone(offset_to_prefetch) && cache->get_pq_occupancy()[0] < cache->PQ_SIZE &&
					cache->get_pq_occupancy()[0] + cache->get_mshr_occupancy() < cache->MSHR_SIZE - 1) {
				uint64_t pf_block_number = block_number + cur_pf_offset;
				uint64_t base_addr = block_number << LOG2_BLOCK_SIZE;
				uint64_t pf_addr = pf_block_number << LOG2_BLOCK_SIZE;
                res.push_back(pf_addr);
				// cache->prefetch_line(0, base_addr, pf_addr, this->pf_level[d], 0);
				// assert(ok == 1);
				this->mark(pf_block_number, MLOP_State::PREFTCH, this->pf_level[d]);
				pf_issued += 1;
			}
		}
	}
	if (this->debug_level >= 2) {
		cout << "[MLOP::prefetch] new_access_map=" << map_to_string(access_map, prefetch_map) << endl;
		cout << "[MLOP::prefetch] issued " << pf_issued << " prefetch(es)" << endl;
	}
    return res;
}

void MLOP::mark(uint64_t block_number, MLOP_State state, int fill_level) {
	this->access_map_table->set_state(block_number, state, fill_level);
}

void MLOP::set_debug_level(int debug_level) {
	this->debug_level = debug_level;
	this->access_map_table->set_debug_level(debug_level);
}

string MLOP::log_offset_scores() {
	Table table(1 + PF_DEGREE, this->offset_scores.size() + 1);
	vector<string> headers = {"Offset"};
	for (uint32_t d = 0; d < PF_DEGREE; d += 1) {
		ostringstream oss;
		oss << "Score[d=" << d + 1 << "]";
		headers.push_back(oss.str());
	}
	table.set_row(0, headers);
	for (uint32_t i = -(this->blocks_in_zone - 1); i <= +(this->blocks_in_zone - 1); i += 1) {
		table.set_cell(i + this->blocks_in_zone, 0, (int)i);
		for (uint32_t d = 0; d < PF_DEGREE; d += 1)
			table.set_cell(i + this->blocks_in_zone, d + 1, (int)this->offset_scores[d][ORIGIN + i]);
	}
	return table.to_string();
}

void MLOP::log() {
	cout << "Access Map Table:" << dec << endl;
	cout << this->access_map_table->log();

	cout << "Offset Scores:" << endl;
	cout << this->log_offset_scores();
}

/*========== stats ==========*/

void MLOP::track(uint64_t block_number) {
	uint64_t zone_number = block_number / this->blocks_in_zone;
	if (this->tracking && zone_number == this->tracked_zone_number) {
		AccessMapTable::Entry *entry = this->access_map_table->find(zone_number);
		if (!entry) {
			this->tracking = false; /* end of zone lifetime, stop tracking */
			this->zone_life.push_back(string(this->blocks_in_zone, state_char[MLOP_State::INIT]));
			return;
		}
		const vector<MLOP_State> &access_map = entry->data.access_map;
		const vector<int> &prefetch_map = entry->data.prefetch_map;
		string s = map_to_string(access_map, prefetch_map);
		if (s != this->zone_life.back())
			this->zone_life.push_back(s);
	}
}

void MLOP::reset_stats() {
	this->tracking = false;
	this->zone_cnt = 0;
	this->zone_life.clear();

	this->round_cnt = 0;
	this->pf_degree_sum = 0;
	this->pf_degree_sqr_sum = 0;
	this->max_score_le_sum = 0;
	this->max_score_le_sqr_sum = 0;
	this->max_score_ri_sum = 0;
	this->max_score_ri_sqr_sum = 0;
}

void MLOP::print_stats() {
	cout << "[MLOP] History of tracked zone:" << endl;
	for (auto &x : this->zone_life)
		cout << x << endl;

	double pf_degree_mean = 1.0 * this->pf_degree_sum / this->round_cnt;
	double pf_degree_sqr_mean = 1.0 * this->pf_degree_sqr_sum / this->round_cnt;
	double pf_degree_sd = sqrt(pf_degree_sqr_mean - square(pf_degree_mean));
	cout << "[MLOP] Prefetch Degree Mean: " << pf_degree_mean << endl;
	cout << "[MLOP] Prefetch Degree SD: " << pf_degree_sd << endl;

	double max_score_le_mean = 1.0 * this->max_score_le_sum / this->round_cnt;
	double max_score_le_sqr_mean = 1.0 * this->max_score_le_sqr_sum / this->round_cnt;
	double max_score_le_sd = sqrt(max_score_le_sqr_mean - square(max_score_le_mean));
	cout << "[MLOP] Max Score Left Mean (%): " << 100.0 * max_score_le_mean / NUM_UPDATES << endl;
	cout << "[MLOP] Max Score Left SD (%): " << 100.0 * max_score_le_sd / NUM_UPDATES << endl;

	double max_score_ri_mean = 1.0 * this->max_score_ri_sum / this->round_cnt;
	double max_score_ri_sqr_mean = 1.0 * this->max_score_ri_sqr_sum / this->round_cnt;
	double max_score_ri_sd = sqrt(max_score_ri_sqr_mean - square(max_score_ri_mean));
	cout << "[MLOP] Max Score Right Mean (%): " << 100.0 * max_score_ri_mean / NUM_UPDATES << endl;
	cout << "[MLOP] Max Score Right SD (%): " << 100.0 * max_score_ri_sd / NUM_UPDATES << endl;
	cout << endl;
}

void MLOP::dump_stats()
{
	print_stats();
}

/* Base-class virtual function */
void MLOP::invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, std::vector<uint64_t> &pref_addr)
{
    uint64_t block_number = address >> LOG2_BLOCK_SIZE;

    /* check prefetch hit */
    bool prefetch_hit = false;
    if (cache_hit == 1) {
        uint32_t set = parent->get_set(block_number);
        uint32_t way = parent->get_way(block_number, set);
        if (parent->block[set * parent->NUM_WAY + way].prefetch == 1)
            prefetch_hit = true;
    }

    /* check trigger access */
    bool trigger_access = false;
    if (cache_hit == 0 || prefetch_hit)
        trigger_access = true;

    if (trigger_access)
        /* update MLOP with most recent trigger access */
        access(block_number);

    /* issue prefetches */
    mark(block_number, MLOP_State::ACCESS);
    pref_addr = prefetch(parent, block_number);

    if (knob::mlop_debug_level >= 3) {
        log();
        cout << "=======================================" << dec << endl;
    }

    /* stats */
    track(block_number);
}

void MLOP::invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, std::vector<int64_t> &offsets)
{
    uint64_t block_number = address >> LOG2_BLOCK_SIZE;

    /* check prefetch hit */
    bool prefetch_hit = false;
    if (cache_hit == 1) {
        uint32_t set = parent->get_set(block_number);
        uint32_t way = parent->get_way(block_number, set);
        if (parent->block[set * parent->NUM_WAY + way].prefetch == 1)
            prefetch_hit = true;
    }

    /* check trigger access */
    bool trigger_access = false;
    if (cache_hit == 0 || prefetch_hit)
        trigger_access = true;

    if (trigger_access)
        /* update MLOP with most recent trigger access */
        access(block_number);

    /* issue prefetches */
    mark(block_number, MLOP_State::ACCESS);
    offsets = prefetch_offsets(parent, block_number);

    if (knob::mlop_debug_level >= 3) {
        log();
        cout << "=======================================" << dec << endl;
    }

    /* stats */
    track(block_number);
}

void MLOP::register_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr)
{
	if (parent->block[set * parent->NUM_WAY + way].valid == 0)
		return; /* no eviction */

	uint64_t evicted_block_number = evicted_addr >> LOG2_BLOCK_SIZE;
	mark(evicted_block_number, MLOP_State::INIT);

	/* stats */
	track(evicted_block_number);
}


# endif /* MLOP_H */
