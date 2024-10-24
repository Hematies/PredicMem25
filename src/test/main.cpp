#include <iostream>
#include "global.hpp"

int main()
{
    // Data:
    static InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> inputBufferEntries[IB_NUM_SETS][IB_NUM_WAYS];
    static DictionaryEntry<delta_t, dic_confidence_t> diccionaryEntries[NUM_CLASSES];
    static svm_weight_t weightMatrix[NUM_CLASSES][SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL];
    static svm_distance_t interceptMatrix[NUM_CLASSES];

    // Prefetcher:
    static GASP gasp = GASP(inputBufferEntries, diccionaryEntries, weightMatrix, interceptMatrix);

    // Test:
    block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE];
    int prefetchDegree = 0;
    
    int i = 0;
    while (true) {
        prefetchDegree = gasp(1, i << 0, addressesToPrefetch);
        prefetchDegree = gasp(2, i << 1, addressesToPrefetch);
        prefetchDegree = gasp(3, i << 2, addressesToPrefetch);
        prefetchDegree = gasp(4, i << 3, addressesToPrefetch);
        prefetchDegree = gasp(5, i << 4, addressesToPrefetch);
        i++;
    }


    std::cout << "Hello World!\n";
}
