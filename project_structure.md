# 📁 part-4-project - Project Structure

part-4-project/
├── 🟡 🚫 **.gitignore**
├── 📂 code/
│ ├── 📂 Final - Systolic Array (optimised)/
│ │ ├── 📂 demo/
│ │ │ ├── 📄 demo.py
│ │ │ ├── 📄 master_data.mif
│ │ │ └── 📄 master_weight.mif
│ │ ├── ⚙️ imagenet_class_index.json
│ │ ├── 📂 MATLAB/
│ │ │ ├── 📄 mif_reader.m
│ │ │ ├── 📄 mif_to_1d_array_matlab.m
│ │ │ └── 📄 simple_packing.m
│ │ ├── 📂 mif/
│ │ │ ├── 📂 original/
│ │ │ ├── 📂 pipeline_v1/
│ │ │ └── 📂 pipeline_v2/
│ │ ├── 📂 Python/
│ │ │ ├── 🖼️ cat.jpg
│ │ │ ├── 🖼️ hand_xray.jpg
│ │ │ ├── 📂 helpers/
│ │ │ │ ├── 📄 corrected_output.py
│ │ │ │ ├── 📄 get_tiles.py
│ │ │ │ ├── 📄 latency_test.py
│ │ │ │ ├── 📄 MAC_calculator.py
│ │ │ │ ├── 📄 pre_processing_strip.py
│ │ │ │ └── 📄 sparsity_plot.py
│ │ │ ├── ⚙️ imagenet_class_index.json
│ │ │ ├── 🖼️ patella_alta.jpg
│ │ │ ├── 📄 pico.py
│ │ │ ├── 📄 pipelined_v1.py
│ │ │ ├── 📄 pipelined_v2.py
│ │ │ └── 📂 testing/
│ │ │ │ ├── 📄 pipelined_v1_random_testing.py
│ │ │ │ ├── 📄 pipelined_v1_structured_testing.py
│ │ │ │ ├── 📄 pipelined_v2_testing.py
│ │ │ │ └── 📄 testing.py
│ │ ├── 📂 quartus/
│ │ │ ├── ALL.bdf
│ │ │ ├── 📄 de1_soc_top.vhd
│ │ │ ├── 📄 DE1_SoC.qsf
│ │ │ ├── 📄 debouncer.vhd
│ │ │ ├── 📄 RX.vhd
│ │ │ ├── 📄 TX.vhd
│ │ │ ├── 📄 UART.vhd
│ │ │ ├── 📄 data_rom.vhd
│ │ │ ├── 📄 weight_rom.vhd
│ │ │ ├── 📄 npu_constraints.sdc
│ │ │ ├── 📄 final.qpf (Quartus project file)
│ │ │ ├── 📄 final.qsf (Quartus settings file)
│ │ │ ├── 📂 db/ (compilation database - build artifacts)
│ │ │ ├── � incremental_db/ (incremental compilation data)
│ │ │ ├── � output_files/ (generated bitstreams and reports)
│ │ │ ├── � greybox_tmp/ (temporary files)
│ │ │ └── � my_soc_system/ (SoC system files)
│ │ ├── 📄 requirements.txt
│ │ ├── 📂 Systolic Array/
│ │ │ ├── 📄 control_unit.vhd
│ │ │ ├── 📄 custom_types.vhd
│ │ │ ├── 📄 npu_wrapper.vhd
│ │ │ ├── 📄 processing_element.vhd
│ │ │ ├── 📄 systolic_array.vhd
│ │ │ ├── 📄 tb_memory.vhd
│ │ │ ├── 📄 tb_npu_wrapper.vhd
│ │ │ ├── 📄 tb_row_stripping_sparsity.vhd
│ │ │ ├── 📄 top_level_systolic_array.vhd
│ │ ├── 📂 testing/
│ │ │ ├── 📂 quartus_timing_analysis/
│ │ │ │ ├── 📂 de1_soc_top/
│ │ │ │ │ ├── 📂 100MHz/
│ │ │ │ │ └── 📂 50MHz/
│ │ │ │ └── 📂 npu_wrapper/
│ │ │ │ │ ├── 📂 m_2/
│ │ │ │ │ └── 📂 m_8/
│ │ │ ├── 📂 v1 \_structured_alexnet/
│ │ │ │ ├── 📂 12.5/
│ │ │ │ ├── 📂 25/
│ │ │ │ ├── 📂 37.5/
│ │ │ │ ├── 📂 50/
│ │ │ │ ├── 📂 62.5/
│ │ │ │ ├── 📂 75/
│ │ │ │ └── 📂 87.5/
│ │ │ ├── 📂 v1_random/
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
│ │ │ │ └── 📄 random_results.xlsx
│ │ │ └── 📂 v2_alexnet/
│ │ │ │ ├── 📂 run_1/
│ │ │ │ │ ├── 📂 tile_0/
│ │ │ │ │ ├── 📂 tile_1/
│ │ │ │ │ ├── 📂 tile_2/
│ │ │ │ │ ├── 📂 tile_3/
│ │ │ │ │ ├── 📂 tile_4/
│ │ │ │ │ ├── 📂 tile_5/
│ │ │ │ │ ├── 📂 tile_6/
│ │ │ │ │ └── 📂 tile_7/
│ │ │ │ ├── 📄 run_1_terminal_output.txt
│ │ │ │ ├── 📂 run_2/
│ │ │ │ │ ├── 📂 tile_0/
│ │ │ │ │ ├── 📂 tile_1/
│ │ │ │ │ ├── 📂 tile_2/
│ │ │ │ │ ├── 📂 tile_3/
│ │ │ │ │ ├── 📂 tile_4/
│ │ │ │ │ ├── 📂 tile_5/
│ │ │ │ │ ├── 📂 tile_6/
│ │ │ │ │ └── 📂 tile_7/
│ │ │ │ ├── 📄 run_2_terminal_output.txt
│ │ │ │ └── 📄 v2_alexnet_results.xlsx
│ ├── 📂 Systolic Array (basic)/
│ │ ├── 📄 array_length_tb.vhd
│ │ ├── 📄 matrix_type.vhd
│ │ ├── 🖼️ Multiplication-of-3-by-3-Matrices-01.png
│ │ ├── 📄 processing_element.vhd
│ │ ├── 🖼️ systolic_array_diagram.jpg
│ │ ├── 📄 systolic_array_tb.vhd
│ │ ├── 📄 systolic_array.vhd
│ │ └── 🖼️ youtube_ss.png
│ └── 📂 Systolic Array (power-gating)/
│ │ ├── 📄 control_unit.vhd
│ │ ├── 📄 custom_types.vhd
│ │ ├── 📄 pe_output.m
│ │ ├── 📄 processing_element.vhd
│ │ ├── 📄 systolic_array.vhd
│ │ ├── 📄 tb_buffer.vhd
│ │ ├── 📄 tb_systolic_array.vhd
│ │ ├── 📄 tb_top_level_systolic_array.vhd
│ │ ├── 📄 top_level_systolic_array.vhd
│ │ ├── 📄 vsim.wlf
├── 📂 figures/
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
├── 📂 general/
│ ├── 📄 Meeting Minutes.docx
│ ├── 📄 Project_43_Seminar_Slides.pptx
│ └── 📄 Resource_list.xlsx
├── 📂 Kristelle/
│ ├── 🖼️ date.png
│ ├── 📄 final_report.bib
│ ├── 📕 P4P Literature Review Mind Map.pdf
│ ├── 📕 P4P_Kristelle_Sampang_Mid_Year_Report.pdf
│ ├── 📕 P4P_Kristelle_Sampang_Project_Scope_Objectives_Literature_Review_Report_v2.pdf
│ ├── 📄 P4P_Kristelle_Sampang_Project_Scope_Objectives_Literature_Review_Report.docx
│ ├── 📕 P4P_Kristelle_Sampang_Project_Scope_Objectives_Literature_Review_Report.pdf
│ ├── 📕 Project-Based Risk Assessment_2025-signed.pdf
│ └── 🖼️ signature.png
├── 📂 Pratham/
├── 📄 projectStructure.exclusions
├── 📂 results/
│ ├── 📂 baseline_modelsim/
│ ├── 🖼️ Effect of M value on the number of Active PEs in the Systolic Array and Latency v2.png
│ ├── 🖼️ Effect of M value on the number of Active PEs in the Systolic Array and Latency.png
│ ├── 🖼️ fmax.png
│ ├── 🖼️ Impact of Stripping Algorithm on Clock Cycles when Applied on Random Sparsity Graph.png
│ ├── 🖼️ Impact of Stripping Algorithm on Clock Cycles when Applied on Random Sparsity.png
│ ├── 🖼️ Impact on Latency as Row Stripping Increases in a 8x8 Activation Graph.png
│ ├── 🖼️ Impact on Latency as Row Stripping Increases in a 8x8 Activation.png
│ ├── 🖼️ Latency of Baseline Systolic Array on Different NxN sizes.png
│ ├── 🖼️ Latency of Baseline Systolic Array on Varying Input Matrix Size.png
│ ├── 📂 optimised/
│ └── 🖼️ Performance Comparison - Average Clock Cycles per Tile of an AlexNet Convolutional Layer.png

```

```
