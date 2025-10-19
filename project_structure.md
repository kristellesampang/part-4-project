# 📁 part-4-project - Project Structure

## Project Details

- Project 43: Reconfigurable Neural Processing Unit (NPU) for Energy-Efficient AI at the Edge
- Students:
  - Kristelle Sampang (ksam836@aucklanduni.ac.nz)
  - Pratham Chhabria (pchh520@aucklanduni.ac.nz)
- Supervisor: Dr Morteza Biglari-Abhari
- Co-Supervisor: Dr Maryam Hemmati

## 1. Project Overview

This project focuses on the design and implementation of a systolic array architecture for efficient neural network processing, particularly targeting optimisations for sparse data handling. The repository is structured to facilitate development, testing, and documentation of the various components involved in the project.

## 2. Features

- **NPU:**
  - Hardware-accelerated matrix multiplication using a systolic array architecture.
  - Configurable matrix dimensions (up to `N x N`, where `N` is defined in `custom_types.all`).
  - Input data and weights loaded from on-chip ROMs.
  - Pipelined data loading to mitigate timing issues.
  - Dedicated row/column counters for efficient matrix indexing, avoiding costly division/modulo.
- **UART:**
  - Full-duplex serial communication (Transmit and Receive).
  - Configurable baud rate (currently targeting 9600 bps at 50 MHz system clock).
  - Robust input synchronizers for asynchronous `RX_LINE`.
  - Pipelined internal logic for timing closure.
- **System Integration:**
  - A state machsine (`npu_wrapper`) controls the NPU operation, including parameter loading, matrix loading, execution, and result retrieval.
  - Dedicated ROM modules for storing NPU input data and weights.
  - Debounced pushbutton input for system `start` (or similar control).

## 3. Design Details

### 3.1. NPU (Neural Processing Unit)

- **File:** `npu_wrapper.vhd` (top-level NPU control), `top_level_systolic_array.vhd` (systolic array core), `processing_element.vhd` (individual processing element).
- **Purpose:** Orchestrates the loading of matrix `A` (data) and matrix `B` (weights) from ROMs, feeds them into the systolic array, and manages the execution flow.
- **Key Optimisations:**
  - **Pipelined ROM Access:** Reads from `data_rom` and `weight_rom` are explicitly pipelined (1-cycle latency) to handle memory access speeds.
  - **Dedicated Row/Column Counters:** Instead of using division and modulo for `(rom_addr - offset) / N` and `(rom_addr - offset) mod N`, which are computationally expensive, the `npu_wrapper` maintains `current_row_idx` and `current_col_idx` as separate, incrementing counters. This significantly improves timing.
  - **State Machine (`fsm_proc`):** Manages parameter loading, matrix loading, NPU execution, and result collection.
- **Parameters:** `N` (matrix dimension, e.g., 8 for 8x8 matrices) defined in `custom_types.all`.

### 3.2. UART (Universal Asynchronous Receiver/Transmitter)

- **Files:** `UART.vhd` (top-level UART), `TX.vhd` (transmit module), `RX.vhd` (receive module).
- **Baud Rate:** Configured for 9600 bps, operating from a 50 MHz system clock.
- **Key Optimisations:**
  - **Pipelined Prescalers:** The `PRSCL` counters (`prscl_count`) within `TX.vhd` and `RX.vhd` are explicitly pipelined by registering their rollover conditions (`prscl_rollover_reg`). This breaks down the long combinatorial paths involved in wide counter increments and comparisons across multiple clock cycles.
  - **Input Synchronizer:** `RX.vhd` includes a two-stage synchronizer (`rx_line_sync_d1`, `rx_line_sync_d2`) for the asynchronous `UART_RXD` input, crucial for mitigating metastability.
  - **Synchronous Resets:** All internal state elements in `TX` and `RX` are reset synchronously with the `RESET` signal.

### 3.3. Top-Level Integration (`de1_soc_top.vhd`)

- **File:** `de1_soc_top.vhd` (or your actual top-level file name).
- **Purpose:** Instantiates the `npu_wrapper`, `UART`, `debouncer`, and any other top-level peripherals. Connects system clocks, resets, and I/O.
- **Crucial:** This is where the clock constraint (`npu_constraints.sdc`) must accurately target the clock input port.

### 3.4. ROMs (Read-Only Memories)

- **Files:** `data_rom.vhd`, `weight_rom.vhd` (or your actual ROM file names).
- **Purpose:** Store the input matrix `A` data and matrix `B` weights for the NPU.
- **Format:** Typically 8-bit data, addressed sequentially. The first few addresses (0, 1, 2) store `active_m`, `active_n`, `active_k` parameters, followed by matrix data.

### 3.5. Debouncer

- **File:** `debouncer.vhd`.
- **Purpose:** Filters noisy pushbutton inputs (`START_BTN`) to produce a clean, single-pulse `start` signal for the NPU.
- **Parameter:** `DEBOUNCE_CYCLES` (e.g., 1_000_000 for 20ms debounce at 50MHz).

## 4. Hardware Requirements

- **FPGA Board:** [Terasic DE1-SoC]
- **FPGA Device:** [Intel Cyclone V 5CSEMA5F31C6N]
- **Clock Source:** 50 MHz clock input.
- **I/O:**
  - One pushbutton for `START`.
  - Two LEDs for visual feedback (e.g., `done` status, `reset` status).
  - UART Transmit (`UART_TXD`) and Receive (`UART_RXD`) pins.
- **Raspberry Pi Pico or Equivalent Microcontroller:** For interfacing and testing UART communication.

## 5. Software Requirements

- **FPGA Synthesis:** Quartus Prime 18.1
- **Simulation:** ModelSim Altera 10.1d
- **Serial Terminal:** For communicating with the UART (e.g., PuTTY, Tera Term, minicom).
- **Python**: (from requirements.txt) torch, torchvision, numpy, Pillow, requests, pyserial
- **MATLAB:** Latest version.

## 6. Setup and Compilation

### 6.1. Project Structure

```
part-4-project/ # Root Directory
├── 🟡 🚫 **.gitignore**
├── 📂 code/ # Source code for the project
│ ├── 📂 Final - Systolic Array (optimised)/ # Optimised implementation of the systolic array
│ │ ├── 📂 demo/ # For Demo
│ │ │ ├── 📄 demo.py
│ │ │ ├── 📄 master_data.mif
│ │ │ └── 📄 master_weight.mif
│ │ ├── ⚙️ imagenet_class_index.json # ImageNet class index for Python scripts
│ │ ├── 📂 MATLAB/ # MATLAB scripts for data processing
│ │ │ ├── 📄 mif_reader.m
│ │ │ ├── 📄 mif_to_1d_array_matlab.m
│ │ │ └── 📄 simple_packing.m
│ │ ├── 📂 mif/ # Contains all the MIF generated
│ │ │ ├── 📂 original/ # Original Version
│ │ │ ├── 📂 pipeline_v1/ # MIFs from pipeline_v1.py
│ │ │ └── 📂 pipeline_v2/ # MIFs from pipeline_v2.py
│ │ ├── 📂 Python/ # Contains all the software co-design files
│ │ │ ├── 🖼️ cat.jpg # Input image for testing
│ │ │ ├── 🖼️ hand_xray.jpg # Input image for testing
│ │ │ ├── 📂 helpers/ # Helper functions to used to create the full software co-design into a single script
│ │ │ │ ├── 📄 corrected_output.py # Post-Processing
│ │ │ │ ├── 📄 get_tiles.py  # Tile extraction
│ │ │ │ ├── 📄 latency_test.py # Calculates theoretical latency
│ │ │ │ ├── 📄 MAC_calculator.py # MAC calculation
│ │ │ │ ├── 📄 pre_processing_strip.py # Pre-processing
│ │ │ │ └── 📄 sparsity_plot.py # Sparsity plotting from AlexNet
│ │ │ ├── ⚙️ imagenet_class_index.json # ImageNet class index for Python scripts
│ │ │ ├── 🖼️ patella_alta.jpg # Input image for testing
│ │ │ ├── 📄 pico.py # Python script running on the Raspberry Pi Pico
│ │ │ ├── 📄 pipelined_v1.py # First Version of the Software Co-Design
│ │ │ ├── 📄 pipelined_v2.py # Second and Final Version of the Software Co-Design
│ │ │ └── 📂 testing/ # Modified pipelined.py code used for testing
│ │ │ │ ├── 📄 pipelined_v1_random_testing.py
│ │ │ │ ├── 📄 pipelined_v1_structured_testing.py
│ │ │ │ ├── 📄 pipelined_v2_testing.py
│ │ │ │ ├── 📄 pipelined_v3_testing.py
│ │ │ │ └── 📄 testing.py
│ │ ├── 📂 quartus/ # Contains the entire Quartus Project
│ │ │ ├── ALL.bdf
│ │ │ ├── 📄 de1_soc_top.vhd # Top-Level
│ │ │ ├── 📄 DE1_SoC.qsf # QSF for pin-assignments
│ │ │ ├── 📄 debouncer.vhd # Debouncer
│ │ │ ├── 📄 RX.vhd # Receiver
│ │ │ ├── 📄 TX.vhd # Transmitter
│ │ │ ├── 📄 UART.vhd # UART Controller
│ │ │ ├── 📄 data_rom.vhd # Data ROM
│ │ │ ├── 📄 weight_rom.vhd # Weight ROM
│ │ │ ├── 📄 npu_constraints.sdc # NPU Constraints, set at 100MHz
│ │ │ ├── 📄 final.qpf (Quartus project file)
│ │ │ ├── 📄 final.qsf (Quartus settings file)
│ │ │ ├── 📂 db/ (compilation database - build artifacts)
│ │ │ ├── 📂 incremental_db/ (incremental compilation data)
│ │ │ ├── 📂 output_files/ (generated bitstreams and reports)
│ │ │ ├── 📂 greybox_tmp/ (temporary files)
│ │ │ └── 📂 my_soc_system/ (SoC system files)
│ │ ├── 📄 requirements.txt # Contains the requirements to run the Python Scripts
│ │ ├── 📂 Systolic Array/ # Hardware Architecture of the NPU
│ │ │ ├── 📄 control_unit.vhd # Control Unit
│ │ │ ├── 📄 custom_types.vhd # Custom Types for this project
│ │ │ ├── 📄 npu_wrapper.vhd # NPU Wrapper
│ │ │ ├── 📄 processing_element.vhd # Processing Element
│ │ │ ├── 📄 systolic_array.vhd # Systolic Array
│ │ │ ├── 📄 tb_memory.vhd # Testbench for Memory
│ │ │ ├── 📄 tb_npu_wrapper.vhd # Testbench for NPU Wrapper
│ │ │ ├── 📄 tb_row_stripping_sparsity.vhd # Testbench for algorithm
│ │ │ ├── 📄 top_level_systolic_array.vhd # Testbench for Control Unit and Systolic Array
│ │ ├── 📂 testing/ # This folder contains all the testing files, including screenshots, HTMs, and the MIF files associated with the particular test case
│ │ │ ├── 📂 quartus_timing_analysis/ # Compilation Reports exported as HTML
│ │ │ │ ├── 📂 de1_soc_top/ # Includes connection to the peripherals and GPIO pins
│ │ │ │ │ ├── 📂 100MHz/ # Synthesised with 100MHz timing constraint
│ │ │ │ │ └── 📂 50MHz/ # Synthesised with 50MHz timing constraint
│ │ │ │ └── 📂 npu_wrapper/ # Only includes top_level_systolic array and the ROMs
│ │ │ │ │ ├── 📂 m_2/ # Synthesised when M = 2
│ │ │ │ │ └── 📂 m_8/ # Synthesised when M = 8
│ │ │ ├── 📂 v1 \_structured_alexnet/ # Test Results from pipelined_v1.py, where each folder name is the sparsity percentage
│ │ │ │ ├── 📂 12.5/
│ │ │ │ ├── 📂 25/
│ │ │ │ ├── 📂 37.5/
│ │ │ │ ├── 📂 50/
│ │ │ │ ├── 📂 62.5/
│ │ │ │ ├── 📂 75/
│ │ │ │ └── 📂 87.5/
│ │ │ ├── 📂 v1_random/ # Test Results from pipelined_v1.py with random sparsity, where each folder name is the sparsity percentage
│ │ │ │ ├── 📂 0/
│ │ │ │ ├── 📂 10/
│ │ │ │ ├── 📂 20/
│ │ │ │ ├── 📂 30/
│ │ │ │ ├── 📂 40/
│ │ │ │ ├── 📂 50/
│ │ │ │ ├── 📂 60/
│ │ │ │ ├── 📂 70/
│ │ │ │ ├── 📂 75/
│ │ │ │ ├── 📂 80/
│ │ │ │ ├── 📂 85/
│ │ │ │ ├── 📂 90/
│ │ │ │ ├── 📂 95/
│ │ │ │ └── 📄 random_results.xlsx # Excel Sheet summarising all results for pipelined_v1.py
│ │ │ └── 📂 v2_alexnet/ # Test Results from pipelined_v2.py
│ │ │ │ ├── 📂 run_1/ # Run 1: XRAY Image
│ │ │ │ │ ├── 📂 tile_0/
│ │ │ │ │ ├── 📂 tile_1/
│ │ │ │ │ ├── 📂 tile_2/
│ │ │ │ │ ├── 📂 tile_3/
│ │ │ │ │ ├── 📂 tile_4/
│ │ │ │ │ ├── 📂 tile_5/
│ │ │ │ │ ├── 📂 tile_6/
│ │ │ │ │ └── 📂 tile_7/
│ │ │ │ ├── 📄 run_1_terminal_output.txt # Python's output terminal for run 1
│ │ │ │ ├── 📂 run_2/ # Run 2: Cat Image
│ │ │ │ │ ├── 📂 tile_0/
│ │ │ │ │ ├── 📂 tile_1/
│ │ │ │ │ ├── 📂 tile_2/
│ │ │ │ │ ├── 📂 tile_3/
│ │ │ │ │ ├── 📂 tile_4/
│ │ │ │ │ ├── 📂 tile_5/
│ │ │ │ │ ├── 📂 tile_6/
│ │ │ │ │ └── 📂 tile_7/
│ │ │ │ ├── 📄 run_2_terminal_output.txt # Python's output terminal for run 2
│ │ │ │ └── 📄 v2_alexnet_results.xlsx # Excelheet summarising all results  for pipelined_v2.py
│ ├── 📂 Systolic Array (basic)/ # Baseline systolic array made in sem1#
│ │ ├── 📄 array_length_tb.vhd # To check array sizes
│ │ ├── 📄 matrix_type.vhd # old types file before custom_types
│ │ ├── 🖼️ Multiplication-of-3-by-3-Matrices-01.png
│ │ ├── 📄 processing_element.vhd # standard PE
│ │ ├── 🖼️ systolic_array_diagram.jpg
│ │ ├── 📄 systolic_array_tb.vhd # for testing the instantiation and math for systolic array
│ │ ├── 📄 systolic_array.vhd # old systolic array code that had shift reg logic only
│ │ └── 🖼️ youtube_ss.png
│ └── 📂 Systolic Array (power-gating)/ # Systolic Array with power gating (current baseline)
│ │ ├── 📄 control_unit.vhd # CU for power-gating, contains the masking to generat the en bit for systolic array PEs
│ │ ├── 📄 custom_types.vhd # Unchanged custom_types file: constains all the MACROS and shortcuts for various VHDL types
│ │ ├── 📄 pe_output.m # matrix output from matlab for cross checking
│ │ ├── 📄 processing_element.vhd # PE with enable bit
│ │ ├── 📄 systolic_array.vhd # Systolic array for PE instantiation
│ │ ├── 📄 tb_buffer.vhd  # TB for simulating the memory read and stagger logic
│ │ ├── 📄 tb_systolic_array.vhd # TB for testing the instantiation and math for systolic array calc
│ │ ├── 📄 tb_top_level_systolic_array.vhd # new TB for modular NPU component approach
│ │ ├── 📄 top_level_systolic_array.vhd # structural entity for holding the NPU component entities
│ │ ├── 📄 vsim.wlf
├── 📂 figures/ # Contains figures and diagrams of system architecture
│ ├── 🖼️ baseline_systolic_array_dataflow.jpg
│ ├── 🖼️ baseline_systolic_array.jpg
│ ├── 🖼️ cat.jpg
│ ├── 🖼️ date.png
│ ├── 🖼️ fsm_npu.jpg
│ ├── 🖼️ MAC.jpg
│ ├── 🖼️ rtl.jpg
│ ├── 🖼️ signature.png
│ ├── 🖼️ system_diagram.jpg
│ ├── 🖼️ tb_fsm.jpg
│ └── 🖼️ UoA-Logo-Primary-RGB-Large.png
├── 📂 general/ # Admin documents that kept track of the project's progress
│ ├── 📄 Meeting Minutes.docx # Meeting minutes for project discussions
│ ├── 📄 Project_43_Seminar_Slides.pptx # Selides presented in the conference seminar
│ └── 📄 Resource_list.xlsx # List of papers found
├── 📂 Kristelle/ # Kristelle's documents including literature review, mid-year report, and final-report
│ ├── 🖼️ date.png
│ ├── 📄 final_report.bib
│ ├── 📕 ksam836_P4P.pdf
│ ├── 📕 P4P Literature Review Mind Map.pdf
│ ├── 📕 P4P_Kristelle_Sampang_Mid_Year_Report.pdf
│ ├── 📕 P4P_Kristelle_Sampang_Project_Scope_Objectives_Literature_Review_Report_v2.pdf
│ ├── 📄 P4P_Kristelle_Sampang_Project_Scope_Objectives_Literature_Review_Report.docx
│ ├── 📕 P4P_Kristelle_Sampang_Project_Scope_Objectives_Literature_Review_Report.pdf
│ ├── 📕 Project-Based Risk Assessment_2025-signed.pdf
│ └── 🖼️ signature.png
├── 📂 Pratham/ # Pratha's
│ ├── 📕p4p_midyear_report_pchh520.pdf
│ ├── 📕P4P_MIRO_BOARD.pdf
│ ├── 📕pchh520_Literature_Review (OLD).pdf
│ ├── 📕pchh520_logbook_Sem1.pdf
│ ├── 📕pchh520_logbook_Sem2.pdf
│ ├── 📕pchh520_p4p_final_report.pdf
├── 📄 projectStructure.exclusions
├── 📂 results/ # Graphs and tables summarising results
│ ├── 📂 baseline_modelsim/ # ModelSim and Python Validation of the Baseline Systolic Array Architecture
│ ├── 🖼️ Effect of M value on the number of Active PEs in the Systolic Array and Latency v2.png
│ ├── 🖼️ Effect of M value on the number of Active PEs in the Systolic Array and Latency.png
│ ├── 🖼️ fmax.png
│ ├── 🖼️ Impact of Stripping Algorithm on Clock Cycles when Applied on Random Sparsity Graph.png
│ ├── 🖼️ Impact of Stripping Algorithm on Clock Cycles when Applied on Random Sparsity.png
│ ├── 🖼️ Impact on Latency as Row Stripping Increases in a 8x8 Activation Graph.png
│ ├── 🖼️ Impact on Latency as Row Stripping Increases in a 8x8 Activation.png
│ ├── 🖼️ Latency of Baseline Systolic Array on Different NxN sizes.png
│ ├── 🖼️ Latency of Baseline Systolic Array on Varying Input Matrix Size.png
│ ├── 📂 optimised/ # ModelSim and Python Validation of the Optimised Systolic Array Architecture
│ └── 🖼️ Performance Comparison - Average Clock Cycles per Tile of an AlexNet Convolutional Layer.png


```

```

```

### 6.2. Quartus Prime Setup

1. Open the Quartus Project located in `code/Final - Systolic Array (optimised)/quartus/`.
2. Check that all files are included and the top-level entity is set correctly.
3. Verify pin assignments in the `.qsf` file match your FPGA board.
4. Ensure the `npu_constraints.sdc` file correctly references your top-level clock port name.
5. Compile the project and ensure there are no errors. For any errors, please contact the authors.

### 6.3. Compilation Steps

1.  **Analysis & Synthesis:** `Processing -> Start -> Start Analysis & Synthesis`. Check for VHDL errors.
2.  **Fitter:** `Processing -> Start -> Start Fitter`. This maps your design to the FPGA resources.
3.  **Timing Analysis:** `Tools -> Timing Analyzer -> Update Timing Netlist` then `Reports -> Report Fmax Summary` and `Reports -> Report Timing Closure Recommendations`. **Verify that setup, hold, and minimum pulse width slacks are all positive.**
4.  **Assembler:** `Processing -> Start -> Start Assembler`. Generates the `.sof` programming file.
5.  **Program Device:** `Tools -> Programmer`. Connect your FPGA board, add the `.sof` file, and program the device.

## 7. Usage

### 7.1. Input Data Format (ROMs)

The `data_rom.vhd` and `weight_rom.vhd` files contain initial data. You can modify these files to change the input matrices.

- **Addresses 0-2:**
  - `ADDR 0`: `active_m` (number of rows in matrix A / rows in result C)
  - `ADDR 1`: `active_n` (number of columns in matrix B / columns in result C)
  - `ADDR 2`: `active_k` (number of columns in matrix A / rows in matrix B)
- **Addresses 3-4:** Reserved/padding.
- **Addresses 5 onwards:** Matrix data, stored row-major for matrix `A` followed by row-major for matrix `B`.
  - Example for `N=8`: `matrix_A[0][0]` at addr 5, `matrix_A[0][1]` at addr 6, ..., `matrix_A[7][7]` then `matrix_B[0][0]`, etc.

### 7.2. UART Communication

1.  **Connect:** Connect your FPGA board's `UART_TXD` and `UART_RXD` pins to a USB-to-UART converter, then connect to your computer.
2.  **Open Terminal:** Use a serial terminal program (PuTTY, Tera Term, etc.) with the following settings:
    - **Baud Rate:** 9600
    - **Data Bits:** 8
    - **Stop Bits:** 1
    - **Parity:** None
    - **Flow Control:** None
3.  **Transmit:** Sending an 8-bit byte from the FPGA to the terminal is initiated by the `send` signal and `data_tx` input on the `UART` module (e.g., from your top-level logic or another controller).
4.  **Receive:** Any data sent from the terminal will be received by the FPGA's `UART_RXD` and will eventually appear on the `LAST_RX_data` signal (and potentially your LEDs).

## 8. Timing Analysis and Performance

- **Target Clock:** 100MHz (10ns period).
- **Current Status:**
  - Achieved Fmax: [`289.27 MHz`]
  - Set Up Slack: [`6.543 ns`]
  - Hold Slack: [`0.308 ns`]
  - Minimum pulse width slack: [`3.949 ns`]
- The NPU itself should achieve a high throughput for matrix multiplication after loading. The overall system performance is gated by the NPU's computation time (`npu_cycle_count`) and the UART's relatively slow baud rate.

## 9. Future Work

- **UART FIFO:** Implement transmit and receive FIFOs for buffering data and reducing CPU/FSM overhead.
- **NPU Pipelined Input/Output:** Stream data into/out of the NPU rather than buffering entire matrices.
- **External Memory Interface:** Store larger matrices in external DDR memory instead of on-chip ROMs.
- **Soft Processor Integration:** Control the NPU and UART via a Nios II (Intel) or MicroBlaze (Xilinx) soft processor.
- **Fine-grained sparsity: **ntroduce a fine-grained sparsity handling system that compacts matrices when unstructured sparsity is present.
- **Real-world Application:** Apply the AlexNet to a real-world application like live footage detection from a camera than processing images.
- **DPR:** Explore and investigate the use of Dynamic Partial Reconfiguration (DPR) to be able to switch out different configurations
