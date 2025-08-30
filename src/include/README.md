# PredicMem25 - `include` Directory

This folder contains the main header files for the **PredicMem25** project, which focuses on memory address and burst length prediction using buffers, dictionaries, and machine learning (SVM).

## File Overview

- **bgasp.hpp**  
  Defines the `BGASP` class, which predicts both memory addresses and burst lengths. It integrates burst-aware input buffers, forwarding buffers, confidence buffers, and SVM-based classification for both address and burst prediction.

- **burst_confidence_buffer.hpp**  
  Contains data structures and logic for managing confidence buffers specifically designed for burst predictions. Handles confidence values and burst-related metadata.

- **burst_input_buffer.hpp**  
  Implements the burst input buffer, which stores sequences of burst lengths and related metadata to support burst prediction.

- **confidence_buffer.hpp**  
  Provides structures and methods for confidence buffers used in address prediction. Manages confidence levels for predicted memory addresses.

- **config.hpp**  
  Central configuration file. Defines constants, buffer sizes, number of classes, and other global parameters used throughout the project.

- **const_expr.hpp**  
  Contains constant expressions and compile-time utilities to support template metaprogramming and static configuration.

- **data_type.hpp**  
  Defines all custom data types used in the project, such as address types, indices, tags, classes, burst lengths, and confidence types.

- **dictionary.hpp**  
  Implements the dictionary structure for mapping deltas (address differences) and classes, supporting both address and burst prediction logic.

- **forwarding_buffer.hpp**  
  Provides the forwarding buffer implementation, which caches recent sequences to accelerate prediction and reduce misses. Includes both address and burst forwarding logic.

- **gasp.hpp**  
  Defines the basic `GASP` class, which predicts only memory addresses (not bursts). Uses input buffers, forwarding buffers, dictionaries, and SVM for address prediction.

- **global.hpp**  
  Contains global definitions, macros, and shared utilities used across multiple modules.

- **init_data.hpp**  
  Provides functions for initializing matrices, buffers, and other data structures required by the prediction algorithms.

- **input_buffer.hpp**  
  Implements the input buffer for memory addresses, including logic for updating, querying, and managing address sequences.

- **svm.hpp**  
  Contains the implementation and utilities for the Support Vector Machine (SVM) classifier, used for both class and burst prediction.

---

Each file is designed to be modular and reusable, supporting the extension and maintenance of the memory prediction system. The structure separates concerns for address and burst prediction, buffer management, configuration, and machine learning