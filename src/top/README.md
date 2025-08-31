# PredicMem25 - `top` Directory

This folder contains the top-level integration files for the **PredicMem25** project. These files connect the prediction modules and manage the main simulation or synthesis flow.

## File Overview

- **top.cpp**  
  Main implementation file.  
  It provides the top functions: entry points of the different hardware modules, working as their hardware synthesis interfaces. Each function calls functions that describe each hardware module and that are defined in the code under the `include` directory. 
  The main functions implemented are:

  - `DictionaryEntry operateDictionary(...)`  
    Reads or writes entries in the dictionary used for delta/class prediction.

  - `InputBufferEntry operateInputBuffer(...)`  
    Reads or updates the input buffer for memory addresses.

  - `void operateSVM(...)`  
    Trains and predicts using the Support Vector Machine for address classes.

  - `void operateSVMWithNop(...)`  
    SVM prediction with support for "nop" (no operation) control.

  - `void prefetchWithGASP(...)`  
    Runs the GASP prefetcher and returns the block address to prefetch.

  - `void prefetchWithGASPWithNop(...)`  
    GASP prefetching with "nop" control.

  - `void prefetchWithSGASP(...)`  
    Runs the SGASP variant for spatial-based prefetching and returns the block address to prefetch.

  - `void prefetchWithSGASPWithNop(...)`  
    Runs SGASP prefetching with "nop" control.

  - `void prefetchWithSGASPWithNopWithDataflow(...)`  
    Runs dataflow-based SGASP prefetching with "nop" control.

  - `void prefetchWithAXI(...)`  
    Reads data from memory using AXI interface.

  - `void prefetchWithSGASPWithAXI(...)`  
    Prefetches data using a SGASP prefetcher and a master AXI bus interface.

  - `void prefetchWithSGASPWithDataflowWithAXI(...)`  
    Prefetches data using a dataflow-based SGASP and a master AXI bus interface.

  - `void computeBurst(...)`  
    Calculates burst address and length for burst prefetching.

  - `void computeBurstWithNop(...)`  
    Burst calculation with "nop" control.

  - `void prefetchWithAXIBurst(...)`  
    Performs burst prefetching using a given master AXI bus interface.

  - `void prefetchWithBSGASP(...)`  
    Runs the BSGASP variant for spatial-based burst prefetching and returns the burst of block address to prefetch.

  - `void prefetchWithBSGASPWithNop(...)`  
    Runs BGASP burst prefetching with "nop" control.

  - `void prefetchWithBSGASPWithDataflow(...)`  
    Runs dataflow-based BGASP burst prefetching.

  - `void prefetchWithBSGASPWithAXI(...)`  
    BGASP burst prefetching with AXI interface.

  - `void prefetchWithBSGASPWithDataflowWithAXI(...)`  
    Prefetches data using a dataflow-based BSGASP and a master AXI bus interface.

  - `void prefetchWithBSGASPWithNopWithDataflow(...)`  
    Prefetches data using a dataflow-based BSGASP, a "nop" control and a master AXI bus interface.

  - `void prefetchWithBSGASPWithNopWithDataflowForTesting(...)`  
    Runs BGASP burst prefetching for testing purposes (work-in-progress).

- **top.hpp**  
  Header file for the top-level module.  
  Declares main interfaces and includes necessary headers.
