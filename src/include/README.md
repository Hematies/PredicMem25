# PredicMem25 - `include` directory

This folder contains the main header files for the **PredicMem25** project, which focuses on memory address and burst length prediction using buffers, dictionaries, and machine learning (SVM). [The report file](report.pdf) contains a full explanation of all the implementation of this folder, except for the GASP, BSGASP and burst-related structures. These elements are in a work-in-progress state.

## Main classes and structures

- **GASP-based classes (`bgasp.hpp`, `gasp.hpp`)**  
  Main prefetcher classes.  
  - `BGASP`: Computes memory block addresses to prefetch and perdicts burst lengths, using burst-aware buffers and SVMs.
  - `GASP`: Computes memory block addresses to prefetch, using standard buffers and SVMs.

- **Input buffer (`input_buffer.hpp`, `burst_input_buffer.hpp`)**  
  - `InputBufferEntry`, `BurstInputBufferEntry`: Store sequences of classes (and burst lengths) for each buffer slot.
  - `InputBuffer`, `BurstInputBuffer`: Manage reading, writing, and  Least Recently Used (LRU) logic for buffer entries.

- **Forwarding buffer (`forwarding_buffer.hpp`)**  
  - `ForwardingBufferEntry`, `BurstForwardingBufferEntry`: Store recent sequences for fast access.
  - `ForwardingBuffer`, `BurstForwardingBuffer`: Manage forwarding logic for address and burst prediction and prefetch.

- **Confidence buffer (`confidence_buffer.hpp`, `burst_confidence_buffer.hpp`)**  
  - `ConfidenceBufferEntry`, `BurstConfidenceBufferEntry`: Store confidence values for prefetches.
  - `ConfidenceBuffer`, `BurstConfidenceBuffer`: Manage confidence updates and retrieval.

- **Confidence forwarding buffer (`forwarding_buffer.hpp`)**  
  - `ConfidenceForwardingBufferEntry`, `BurstConfidenceForwardingBufferEntry`: Store confidence and prefetching info for forwarding.
  - `ConfidenceForwardingBuffer`, `BurstConfidenceForwardingBuffer`: Manage forwarding of confidence data.

- **Dictionary (`dictionary.hpp`)**  
  - `DictionaryEntry`, `DictionaryEntriesMatrix`: Store delta/class mappings and Least Frequently Used (LFU) confidence.
  - `Dictionary`: Handles reading, writing, and updating LFU confidence for dictionary entries.

- **SVM structures (`svm.hpp`)**  
  - `SVMWholeMatrix`, `BurstSVMWholeMatrix`: Store weights and intercepts for SVM classifiers.
  - `SVM`: Implements training and prediction for both address delta and burst classes.

- **Configuration and data types (`config.hpp`, `data_type.hpp`)**  
  - Defines constants, buffer sizes, and custom types (addresses, deltas, indices, classes, burst lengths, etc.).

- **Initialization utilities (`init_data.hpp`)**  
  - Functions to initialize buffers, dictionaries, and SVM matrices with default values.

- **Global includes (`global.hpp`)**  
  - Aggregates all main headers and provides shared macros and definitions.

- **Constant expressions (`const_expr.hpp`)**  
  - Compile-time utilities and lookup table generation.
