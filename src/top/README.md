# PredicMem25 - `top` Directory

This folder contains the top-level integration files for the **PredicMem25** project. These files connect the prediction modules and manage the main simulation or synthesis flow.

## File Overview

- **top.cpp**  
  Main implementation file.  
  It provides the entry points and orchestration for memory address and burst prediction, including hardware synthesis interfaces.  
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
    Runs the GASP prefetcher for address prediction.

  - `void prefetchWithSGASP(...)`  
    Runs the SGASP variant for region-based address prediction.

  - `void prefetchWithSGASPWithAXI(...)`  
    Prefetches data using SGASP and AXI bus interface.

  - `void prefetchWithAXI(...)`  
    Reads data from memory using AXI interface.

  - `void prefetchWithSGASPWithDataflowWithAXI(...)`  
    Dataflow-oriented SGASP prefetching with AXI.

  - `void computeBurst(...)`  
    Calculates burst address and length for prefetching.

  - `void computeBurstWithNop(...)`  
    Burst calculation with "nop" control.

  - `void prefetchWithAXIBurst(...)`  
    Performs burst prefetching using AXI and HLS streams.

  - `void prefetchWithBSGASP(...)`  
    Runs the BGASP prefetcher for burst-aware prediction.

  - `void prefetchWithBSGASPWithDataflow(...)`  
    Dataflow-oriented BGASP burst prediction.

  - `void prefetchWithBSGASPWithAXI(...)`  
    BGASP burst prediction with AXI interface.

  - `void prefetchWithBSGASPWithDataflowWithAXI(...)`  
    Dataflow BGASP burst prediction with AXI.

  - `void prefetchWithGASPWithNop(...)`  
    GASP prefetching with "nop" control.

  - `void prefetchWithSGASPWithNop(...)`  
    SGASP prefetching with "nop" control.

  - `void prefetchWithSGASPWithNopWithDataflow(...)`  
    Dataflow SGASP prefetching with "nop" control.

  - `void prefetchWithBSGASPWithNop(...)`  
    BGASP burst prediction with "nop" control.

  - `void prefetchWithBSGASPWithNopWithDataflow(...)`  
    Dataflow BGASP burst prediction with "nop" control.

  - `void prefetchWithBSGASPWithNopWithDataflowForTesting(...)`  
    BGASP burst prediction for testing purposes.

- **top.hpp**  
  Header file for the top-level module.  
  Declares main interfaces and includes necessary headers.

---

The `top` directory is the entry point for running experiments and integrating all memory prediction components.