# 📁 part-4-project - Project Structure

_Generated on: 10/19/2025, 9:49:25 PM (Cleaned)_

## 📋 Quick Overview

| Metric                | Value                                 |
| --------------------- | ------------------------------------- |
| �️ Primary Tech Stack | VHDL, Python, MATLAB                  |
| 🎯 Project Type       | Systolic Array Implementation         |
| � Main Directories    | Source code, Documentation, Test data |

## ⭐ Important Files

- � 🚫 **.gitignore** - Git ignore rules
- 📄 **project_structure.md** - This file
- 📄 **requirements.txt** - Python dependencies

## 🌳 Directory Structure

```
part-4-project/
├── 🟡 🚫 **.gitignore**
├── 📂 code/
│   ├── 📂 Final - Systolic Array (optimised)/
│   │   ├── 📂 demo/
│   │   │   ├── 📄 demo.py
│   │   │   ├── 📄 master_data.mif
│   │   │   └── 📄 master_weight.mif
│   │   ├── ⚙️ imagenet_class_index.json
│   │   ├── 📂 MATLAB/
│   │   │   ├── 📄 mif_reader.m
│   │   │   ├── 📄 mif_to_1d_array_matlab.m
│   │   │   └── 📄 simple_packing.m
│   │   ├── 📂 mif/
│   │   │   ├── 📂 original/
│   │   │   │   ├── 📄 activation_tile_0.mif
│   │   │   │   ├── 📄 activation_tile_1.mif
│   │   │   │   ├── 📄 activation_tile_2.mif
│   │   │   │   ├── 📄 activation_tile_3.mif
│   │   │   │   ├── 📄 activation_tile_4.mif
│   │   │   │   ├── 📄 activation_tile_5.mif
│   │   │   │   ├── 📄 activation_tile_6.mif
│   │   │   │   ├── 📄 activation_tile_7.mif
│   │   │   │   ├── 📄 weight_tile_0.mif
│   │   │   │   ├── 📄 weight_tile_1.mif
│   │   │   │   ├── 📄 weight_tile_2.mif
│   │   │   │   ├── 📄 weight_tile_3.mif
│   │   │   │   ├── 📄 weight_tile_4.mif
│   │   │   │   ├── 📄 weight_tile_5.mif
│   │   │   │   ├── 📄 weight_tile_6.mif
│   │   │   │   └── 📄 weight_tile_7.mif
│   │   │   ├── 📂 pipeline_v1/
│   │   │   │   ├── 📄 activation_tile_0.mif
│   │   │   │   ├── 📄 activation_tile_1.mif
│   │   │   │   ├── 📄 activation_tile_2.mif
│   │   │   │   ├── 📄 activation_tile_3.mif
│   │   │   │   ├── 📄 activation_tile_4.mif
│   │   │   │   ├── 📄 activation_tile_5.mif
│   │   │   │   ├── 📄 activation_tile_6.mif
│   │   │   │   ├── 📄 activation_tile_7.mif
│   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   ├── 📄 weight_tile_0.mif
│   │   │   │   ├── 📄 weight_tile_1.mif
│   │   │   │   ├── 📄 weight_tile_2.mif
│   │   │   │   ├── 📄 weight_tile_3.mif
│   │   │   │   ├── 📄 weight_tile_4.mif
│   │   │   │   ├── 📄 weight_tile_5.mif
│   │   │   │   ├── 📄 weight_tile_6.mif
│   │   │   │   └── 📄 weight_tile_7.mif
│   │   │   └── 📂 pipeline_v2/
│   │   │   │   ├── 📄 activation_tile_0.mif
│   │   │   │   ├── 📄 activation_tile_1.mif
│   │   │   │   ├── 📄 activation_tile_2.mif
│   │   │   │   ├── 📄 activation_tile_3.mif
│   │   │   │   ├── 📄 activation_tile_4.mif
│   │   │   │   ├── 📄 activation_tile_5.mif
│   │   │   │   ├── 📄 activation_tile_6.mif
│   │   │   │   ├── 📄 activation_tile_7.mif
│   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   ├── 📄 weight_tile_0.mif
│   │   │   │   ├── 📄 weight_tile_1.mif
│   │   │   │   ├── 📄 weight_tile_2.mif
│   │   │   │   ├── 📄 weight_tile_3.mif
│   │   │   │   ├── 📄 weight_tile_4.mif
│   │   │   │   ├── 📄 weight_tile_5.mif
│   │   │   │   ├── 📄 weight_tile_6.mif
│   │   │   │   └── 📄 weight_tile_7.mif
│   │   ├── 📂 Python/
│   │   │   ├── 🖼️ cat.jpg
│   │   │   ├── 📂 db/
│   │   │   │   ├── 📄 uart.db_info
│   │   │   │   └── 📄 uart.sld_design_entry.sci
│   │   │   ├── 🖼️ hand_xray.jpg
│   │   │   ├── 📂 helpers/
│   │   │   │   ├── 📄 corrected_output.py
│   │   │   │   ├── 📄 get_tiles.py
│   │   │   │   ├── 📄 latency_test.py
│   │   │   │   ├── 📄 MAC_calculator.py
│   │   │   │   ├── 📄 pre_processing_strip.py
│   │   │   │   └── 📄 sparsity_plot.py
│   │   │   ├── ⚙️ imagenet_class_index.json
│   │   │   ├── 🖼️ patella_alta.jpg
│   │   │   ├── 📄 pico.py
│   │   │   ├── 📄 pipelined_v1.py
│   │   │   ├── 📄 pipelined_v2.py
│   │   │   └── 📂 testing/
│   │   │   │   ├── 📄 pipelined_v1_random_testing.py
│   │   │   │   ├── 📄 pipelined_v1_structured_testing.py
│   │   │   │   ├── 📄 pipelined_v2_testing.py
│   │   │   │   └── 📄 testing.py
│   │   ├── 📂 quartus/
│   │   │   ├──  ALL.bdf
│   │   │   ├── 📄 de1_soc_top.vhd
│   │   │   ├── 📄 DE1_SoC.qsf
│   │   │   ├── 📄 debouncer.vhd
│   │   │   ├── 📄 RX.vhd
│   │   │   ├── 📄 TX.vhd
│   │   │   ├── 📄 UART.vhd
│   │   │   ├── 📄 data_rom.vhd
│   │   │   ├── � weight_rom.vhd
│   │   │   ├── 📄 npu_constraints.sdc
│   │   │   ├── 📄 final.qpf (Quartus project file)
│   │   │   ├── 📄 final.qsf (Quartus settings file)
│   │   │   ├── 📂 db/ (compilation database - build artifacts)
│   │   │   ├── � incremental_db/ (incremental compilation data)
│   │   │   ├── � output_files/ (generated bitstreams and reports)
│   │   │   ├── � greybox_tmp/ (temporary files)
│   │   │   └── � my_soc_system/ (SoC system files)
│   │   ├── 📄 requirements.txt
│   │   ├── 📂 Systolic Array/
│   │   │   ├── 📄 control_unit.vhd
│   │   │   ├── 📄 control_unit.vhd.bak
│   │   │   ├── 📄 custom_types.vhd
│   │   │   ├── 📄 custom_types.vhd.bak
│   │   │   ├── 📄 npu_wrapper.vhd
│   │   │   ├── 📄 npu_wrapper.vhd.bak
│   │   │   ├── 📄 processing_element.vhd
│   │   │   ├── 📄 systolic_array.vhd
│   │   │   ├── 📄 tb_memory.vhd
│   │   │   ├── 📄 tb_npu_wrapper.vhd
│   │   │   ├── 📄 tb_row_stripping_sparsity.vhd
│   │   │   ├── 📄 top_level_systolic_array.vhd
│   │   │   └── 📄 top_level_systolic_array.vhd.bak
│   │   ├── 📂 testing/
│   │   │   ├── 📂 quartus_timing_analysis/
│   │   │   │   ├── 📂 de1_soc_top/
│   │   │   │   │   ├── 📂 100MHz/
│   │   │   │   │   │   ├── 📄 final-Analysis - Synthesis.htm
│   │   │   │   │   │   ├── 📂 final-Analysis - Synthesis.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 10.htm
│   │   │   │   │   │   │   ├── 📄 11.htm
│   │   │   │   │   │   │   ├── 📄 12.htm
│   │   │   │   │   │   │   ├── 📄 13.htm
│   │   │   │   │   │   │   ├── 📄 14.htm
│   │   │   │   │   │   │   ├── 📄 15.htm
│   │   │   │   │   │   │   ├── 📄 16.htm
│   │   │   │   │   │   │   ├── 📄 17.htm
│   │   │   │   │   │   │   ├── 📄 18.htm
│   │   │   │   │   │   │   ├── 📄 19.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 20.htm
│   │   │   │   │   │   │   ├── 📄 21.htm
│   │   │   │   │   │   │   ├── 📄 22.htm
│   │   │   │   │   │   │   ├── 📄 23.htm
│   │   │   │   │   │   │   ├── 📄 24.htm
│   │   │   │   │   │   │   ├── 📄 25.htm
│   │   │   │   │   │   │   ├── 📄 26.htm
│   │   │   │   │   │   │   ├── 📄 27.htm
│   │   │   │   │   │   │   ├── 📄 28.htm
│   │   │   │   │   │   │   ├── 📄 29.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 30.htm
│   │   │   │   │   │   │   ├── 📄 31.htm
│   │   │   │   │   │   │   ├── 📄 32.htm
│   │   │   │   │   │   │   ├── 📄 33.htm
│   │   │   │   │   │   │   ├── 📄 34.htm
│   │   │   │   │   │   │   ├── 📄 35.htm
│   │   │   │   │   │   │   ├── 📄 36.htm
│   │   │   │   │   │   │   ├── 📄 37.htm
│   │   │   │   │   │   │   ├── 📄 38.htm
│   │   │   │   │   │   │   ├── 📄 39.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 40.htm
│   │   │   │   │   │   │   ├── 📄 41.htm
│   │   │   │   │   │   │   ├── 📄 42.htm
│   │   │   │   │   │   │   ├── 📄 43.htm
│   │   │   │   │   │   │   ├── 📄 44.htm
│   │   │   │   │   │   │   ├── 📄 45.htm
│   │   │   │   │   │   │   ├── 📄 46.htm
│   │   │   │   │   │   │   ├── 📄 47.htm
│   │   │   │   │   │   │   ├── 📄 48.htm
│   │   │   │   │   │   │   ├── 📄 49.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 50.htm
│   │   │   │   │   │   │   ├── 📄 51.htm
│   │   │   │   │   │   │   ├── 📄 52.htm
│   │   │   │   │   │   │   ├── 📄 53.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 📄 8.htm
│   │   │   │   │   │   │   ├── 📄 9.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Assembler.htm
│   │   │   │   │   │   ├── 📂 final-Assembler.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Fitter.htm
│   │   │   │   │   │   ├── 📂 final-Fitter.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 10.htm
│   │   │   │   │   │   │   ├── 📄 11.htm
│   │   │   │   │   │   │   ├── 📄 12.htm
│   │   │   │   │   │   │   ├── 📄 13.htm
│   │   │   │   │   │   │   ├── 📄 14.htm
│   │   │   │   │   │   │   ├── 📄 15.htm
│   │   │   │   │   │   │   ├── 📄 16.htm
│   │   │   │   │   │   │   ├── 📄 17.htm
│   │   │   │   │   │   │   ├── 📄 18.htm
│   │   │   │   │   │   │   ├── 📄 19.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 20.htm
│   │   │   │   │   │   │   ├── 📄 21.htm
│   │   │   │   │   │   │   ├── 📄 22.htm
│   │   │   │   │   │   │   ├── 📄 23.htm
│   │   │   │   │   │   │   ├── 📄 24.htm
│   │   │   │   │   │   │   ├── 📄 25.htm
│   │   │   │   │   │   │   ├── 📄 26.htm
│   │   │   │   │   │   │   ├── 📄 27.htm
│   │   │   │   │   │   │   ├── 📄 28.htm
│   │   │   │   │   │   │   ├── 📄 29.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 30.htm
│   │   │   │   │   │   │   ├── 📄 31.htm
│   │   │   │   │   │   │   ├── 📄 32.htm
│   │   │   │   │   │   │   ├── 📄 33.htm
│   │   │   │   │   │   │   ├── 📄 34.htm
│   │   │   │   │   │   │   ├── 📄 35.htm
│   │   │   │   │   │   │   ├── 📄 36.htm
│   │   │   │   │   │   │   ├── 📄 37.htm
│   │   │   │   │   │   │   ├── 📄 38.htm
│   │   │   │   │   │   │   ├── 📄 39.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 📄 8.htm
│   │   │   │   │   │   │   ├── 📄 9.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow Elapsed Time.htm
│   │   │   │   │   │   ├── 📂 final-Flow Elapsed Time.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow Non-Default Global Settings.htm
│   │   │   │   │   │   ├── 📂 final-Flow Non-Default Global Settings.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow OS Summary.htm
│   │   │   │   │   │   ├── 📂 final-Flow OS Summary.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow Settings.htm
│   │   │   │   │   │   ├── 📂 final-Flow Settings.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow Summary.htm
│   │   │   │   │   │   ├── 📂 final-Flow Summary.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Timing Analyzer.htm
│   │   │   │   │   │   └── 📂 final-Timing Analyzer.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 10.htm
│   │   │   │   │   │   │   ├── 📄 11.htm
│   │   │   │   │   │   │   ├── 📄 12.htm
│   │   │   │   │   │   │   ├── 📄 13.htm
│   │   │   │   │   │   │   ├── 📄 14.htm
│   │   │   │   │   │   │   ├── 📄 15.htm
│   │   │   │   │   │   │   ├── 📄 16.htm
│   │   │   │   │   │   │   ├── 📄 17.htm
│   │   │   │   │   │   │   ├── 📄 18.htm
│   │   │   │   │   │   │   ├── 📄 19.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 20.htm
│   │   │   │   │   │   │   ├── 📄 21.htm
│   │   │   │   │   │   │   ├── 📄 22.htm
│   │   │   │   │   │   │   ├── 📄 23.htm
│   │   │   │   │   │   │   ├── 📄 24.htm
│   │   │   │   │   │   │   ├── 📄 25.htm
│   │   │   │   │   │   │   ├── 📄 26.htm
│   │   │   │   │   │   │   ├── 📄 27.htm
│   │   │   │   │   │   │   ├── 📄 28.htm
│   │   │   │   │   │   │   ├── 📄 29.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 30.htm
│   │   │   │   │   │   │   ├── 📄 31.htm
│   │   │   │   │   │   │   ├── 📄 32.htm
│   │   │   │   │   │   │   ├── 📄 33.htm
│   │   │   │   │   │   │   ├── 📄 34.htm
│   │   │   │   │   │   │   ├── 📄 35.htm
│   │   │   │   │   │   │   ├── 📄 36.htm
│   │   │   │   │   │   │   ├── 📄 37.htm
│   │   │   │   │   │   │   ├── 📄 38.htm
│   │   │   │   │   │   │   ├── 📄 39.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 40.htm
│   │   │   │   │   │   │   ├── 📄 41.htm
│   │   │   │   │   │   │   ├── 📄 42.htm
│   │   │   │   │   │   │   ├── 📄 43.htm
│   │   │   │   │   │   │   ├── 📄 44.htm
│   │   │   │   │   │   │   ├── 📄 45.htm
│   │   │   │   │   │   │   ├── 📄 46.htm
│   │   │   │   │   │   │   ├── 📄 47.htm
│   │   │   │   │   │   │   ├── 📄 48.htm
│   │   │   │   │   │   │   ├── 📄 49.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 50.htm
│   │   │   │   │   │   │   ├── 📄 51.htm
│   │   │   │   │   │   │   ├── 📄 52.htm
│   │   │   │   │   │   │   ├── 📄 53.htm
│   │   │   │   │   │   │   ├── 📄 54.htm
│   │   │   │   │   │   │   ├── 📄 55.htm
│   │   │   │   │   │   │   ├── 📄 56.htm
│   │   │   │   │   │   │   ├── 📄 57.htm
│   │   │   │   │   │   │   ├── 📄 58.htm
│   │   │   │   │   │   │   ├── 📄 59.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 60.htm
│   │   │   │   │   │   │   ├── 📄 61.htm
│   │   │   │   │   │   │   ├── 📄 62.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 📄 8.htm
│   │   │   │   │   │   │   ├── 📄 9.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   └── 📂 50MHz/
│   │   │   │   │   │   ├── 📄 final-Analysis - Synthesis.htm
│   │   │   │   │   │   ├── 📂 final-Analysis - Synthesis.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 10.htm
│   │   │   │   │   │   │   ├── 📄 11.htm
│   │   │   │   │   │   │   ├── 📄 12.htm
│   │   │   │   │   │   │   ├── 📄 13.htm
│   │   │   │   │   │   │   ├── 📄 14.htm
│   │   │   │   │   │   │   ├── 📄 15.htm
│   │   │   │   │   │   │   ├── 📄 16.htm
│   │   │   │   │   │   │   ├── 📄 17.htm
│   │   │   │   │   │   │   ├── 📄 18.htm
│   │   │   │   │   │   │   ├── 📄 19.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 20.htm
│   │   │   │   │   │   │   ├── 📄 21.htm
│   │   │   │   │   │   │   ├── 📄 22.htm
│   │   │   │   │   │   │   ├── 📄 23.htm
│   │   │   │   │   │   │   ├── 📄 24.htm
│   │   │   │   │   │   │   ├── 📄 25.htm
│   │   │   │   │   │   │   ├── 📄 26.htm
│   │   │   │   │   │   │   ├── 📄 27.htm
│   │   │   │   │   │   │   ├── 📄 28.htm
│   │   │   │   │   │   │   ├── 📄 29.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 30.htm
│   │   │   │   │   │   │   ├── 📄 31.htm
│   │   │   │   │   │   │   ├── 📄 32.htm
│   │   │   │   │   │   │   ├── 📄 33.htm
│   │   │   │   │   │   │   ├── 📄 34.htm
│   │   │   │   │   │   │   ├── 📄 35.htm
│   │   │   │   │   │   │   ├── 📄 36.htm
│   │   │   │   │   │   │   ├── 📄 37.htm
│   │   │   │   │   │   │   ├── 📄 38.htm
│   │   │   │   │   │   │   ├── 📄 39.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 40.htm
│   │   │   │   │   │   │   ├── 📄 41.htm
│   │   │   │   │   │   │   ├── 📄 42.htm
│   │   │   │   │   │   │   ├── 📄 43.htm
│   │   │   │   │   │   │   ├── 📄 44.htm
│   │   │   │   │   │   │   ├── 📄 45.htm
│   │   │   │   │   │   │   ├── 📄 46.htm
│   │   │   │   │   │   │   ├── 📄 47.htm
│   │   │   │   │   │   │   ├── 📄 48.htm
│   │   │   │   │   │   │   ├── 📄 49.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 50.htm
│   │   │   │   │   │   │   ├── 📄 51.htm
│   │   │   │   │   │   │   ├── 📄 52.htm
│   │   │   │   │   │   │   ├── 📄 53.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 📄 8.htm
│   │   │   │   │   │   │   ├── 📄 9.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Assembler.htm
│   │   │   │   │   │   ├── 📂 final-Assembler.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Fitter.htm
│   │   │   │   │   │   ├── 📂 final-Fitter.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 10.htm
│   │   │   │   │   │   │   ├── 📄 11.htm
│   │   │   │   │   │   │   ├── 📄 12.htm
│   │   │   │   │   │   │   ├── 📄 13.htm
│   │   │   │   │   │   │   ├── 📄 14.htm
│   │   │   │   │   │   │   ├── 📄 15.htm
│   │   │   │   │   │   │   ├── 📄 16.htm
│   │   │   │   │   │   │   ├── 📄 17.htm
│   │   │   │   │   │   │   ├── 📄 18.htm
│   │   │   │   │   │   │   ├── 📄 19.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 20.htm
│   │   │   │   │   │   │   ├── 📄 21.htm
│   │   │   │   │   │   │   ├── 📄 22.htm
│   │   │   │   │   │   │   ├── 📄 23.htm
│   │   │   │   │   │   │   ├── 📄 24.htm
│   │   │   │   │   │   │   ├── 📄 25.htm
│   │   │   │   │   │   │   ├── 📄 26.htm
│   │   │   │   │   │   │   ├── 📄 27.htm
│   │   │   │   │   │   │   ├── 📄 28.htm
│   │   │   │   │   │   │   ├── 📄 29.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 30.htm
│   │   │   │   │   │   │   ├── 📄 31.htm
│   │   │   │   │   │   │   ├── 📄 32.htm
│   │   │   │   │   │   │   ├── 📄 33.htm
│   │   │   │   │   │   │   ├── 📄 34.htm
│   │   │   │   │   │   │   ├── 📄 35.htm
│   │   │   │   │   │   │   ├── 📄 36.htm
│   │   │   │   │   │   │   ├── 📄 37.htm
│   │   │   │   │   │   │   ├── 📄 38.htm
│   │   │   │   │   │   │   ├── 📄 39.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 📄 8.htm
│   │   │   │   │   │   │   ├── 📄 9.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow Elapsed Time.htm
│   │   │   │   │   │   ├── 📂 final-Flow Elapsed Time.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow Log.htm
│   │   │   │   │   │   ├── 📂 final-Flow Log.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow Non-Default Global Settings.htm
│   │   │   │   │   │   ├── 📂 final-Flow Non-Default Global Settings.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow OS Summary.htm
│   │   │   │   │   │   ├── 📂 final-Flow OS Summary.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow Settings.htm
│   │   │   │   │   │   ├── 📂 final-Flow Settings.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow Summary.htm
│   │   │   │   │   │   ├── 📂 final-Flow Summary.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Timing Analyzer.htm
│   │   │   │   │   │   └── 📂 final-Timing Analyzer.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 10.htm
│   │   │   │   │   │   │   ├── 📄 11.htm
│   │   │   │   │   │   │   ├── 📄 12.htm
│   │   │   │   │   │   │   ├── 📄 13.htm
│   │   │   │   │   │   │   ├── 📄 14.htm
│   │   │   │   │   │   │   ├── 📄 15.htm
│   │   │   │   │   │   │   ├── 📄 16.htm
│   │   │   │   │   │   │   ├── 📄 17.htm
│   │   │   │   │   │   │   ├── 📄 18.htm
│   │   │   │   │   │   │   ├── 📄 19.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 20.htm
│   │   │   │   │   │   │   ├── 📄 21.htm
│   │   │   │   │   │   │   ├── 📄 22.htm
│   │   │   │   │   │   │   ├── 📄 23.htm
│   │   │   │   │   │   │   ├── 📄 24.htm
│   │   │   │   │   │   │   ├── 📄 25.htm
│   │   │   │   │   │   │   ├── 📄 26.htm
│   │   │   │   │   │   │   ├── 📄 27.htm
│   │   │   │   │   │   │   ├── 📄 28.htm
│   │   │   │   │   │   │   ├── 📄 29.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 30.htm
│   │   │   │   │   │   │   ├── 📄 31.htm
│   │   │   │   │   │   │   ├── 📄 32.htm
│   │   │   │   │   │   │   ├── 📄 33.htm
│   │   │   │   │   │   │   ├── 📄 34.htm
│   │   │   │   │   │   │   ├── 📄 35.htm
│   │   │   │   │   │   │   ├── 📄 36.htm
│   │   │   │   │   │   │   ├── 📄 37.htm
│   │   │   │   │   │   │   ├── 📄 38.htm
│   │   │   │   │   │   │   ├── 📄 39.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 40.htm
│   │   │   │   │   │   │   ├── 📄 41.htm
│   │   │   │   │   │   │   ├── 📄 42.htm
│   │   │   │   │   │   │   ├── 📄 43.htm
│   │   │   │   │   │   │   ├── 📄 44.htm
│   │   │   │   │   │   │   ├── 📄 45.htm
│   │   │   │   │   │   │   ├── 📄 46.htm
│   │   │   │   │   │   │   ├── 📄 47.htm
│   │   │   │   │   │   │   ├── 📄 48.htm
│   │   │   │   │   │   │   ├── 📄 49.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 50.htm
│   │   │   │   │   │   │   ├── 📄 51.htm
│   │   │   │   │   │   │   ├── 📄 52.htm
│   │   │   │   │   │   │   ├── 📄 53.htm
│   │   │   │   │   │   │   ├── 📄 54.htm
│   │   │   │   │   │   │   ├── 📄 55.htm
│   │   │   │   │   │   │   ├── 📄 56.htm
│   │   │   │   │   │   │   ├── 📄 57.htm
│   │   │   │   │   │   │   ├── 📄 58.htm
│   │   │   │   │   │   │   ├── 📄 59.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 60.htm
│   │   │   │   │   │   │   ├── 📄 61.htm
│   │   │   │   │   │   │   ├── 📄 62.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 📄 8.htm
│   │   │   │   │   │   │   ├── 📄 9.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   └── 📂 npu_wrapper/
│   │   │   │   │   ├── 📂 m_2/
│   │   │   │   │   │   ├── 📄 final-Analysis - Synthesis.htm
│   │   │   │   │   │   ├── 📂 final-Analysis - Synthesis.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 10.htm
│   │   │   │   │   │   │   ├── 📄 11.htm
│   │   │   │   │   │   │   ├── 📄 12.htm
│   │   │   │   │   │   │   ├── 📄 13.htm
│   │   │   │   │   │   │   ├── 📄 14.htm
│   │   │   │   │   │   │   ├── 📄 15.htm
│   │   │   │   │   │   │   ├── 📄 16.htm
│   │   │   │   │   │   │   ├── 📄 17.htm
│   │   │   │   │   │   │   ├── 📄 18.htm
│   │   │   │   │   │   │   ├── 📄 19.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 20.htm
│   │   │   │   │   │   │   ├── 📄 21.htm
│   │   │   │   │   │   │   ├── 📄 22.htm
│   │   │   │   │   │   │   ├── 📄 23.htm
│   │   │   │   │   │   │   ├── 📄 24.htm
│   │   │   │   │   │   │   ├── 📄 25.htm
│   │   │   │   │   │   │   ├── 📄 26.htm
│   │   │   │   │   │   │   ├── 📄 27.htm
│   │   │   │   │   │   │   ├── 📄 28.htm
│   │   │   │   │   │   │   ├── 📄 29.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 30.htm
│   │   │   │   │   │   │   ├── 📄 31.htm
│   │   │   │   │   │   │   ├── 📄 32.htm
│   │   │   │   │   │   │   ├── 📄 33.htm
│   │   │   │   │   │   │   ├── 📄 34.htm
│   │   │   │   │   │   │   ├── 📄 35.htm
│   │   │   │   │   │   │   ├── 📄 36.htm
│   │   │   │   │   │   │   ├── 📄 37.htm
│   │   │   │   │   │   │   ├── 📄 38.htm
│   │   │   │   │   │   │   ├── 📄 39.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 40.htm
│   │   │   │   │   │   │   ├── 📄 41.htm
│   │   │   │   │   │   │   ├── 📄 42.htm
│   │   │   │   │   │   │   ├── 📄 43.htm
│   │   │   │   │   │   │   ├── 📄 44.htm
│   │   │   │   │   │   │   ├── 📄 45.htm
│   │   │   │   │   │   │   ├── 📄 46.htm
│   │   │   │   │   │   │   ├── 📄 47.htm
│   │   │   │   │   │   │   ├── 📄 48.htm
│   │   │   │   │   │   │   ├── 📄 49.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 📄 8.htm
│   │   │   │   │   │   │   ├── 📄 9.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Assembler.htm
│   │   │   │   │   │   ├── 📂 final-Assembler.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Fitter.htm
│   │   │   │   │   │   ├── 📂 final-Fitter.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 10.htm
│   │   │   │   │   │   │   ├── 📄 11.htm
│   │   │   │   │   │   │   ├── 📄 12.htm
│   │   │   │   │   │   │   ├── 📄 13.htm
│   │   │   │   │   │   │   ├── 📄 14.htm
│   │   │   │   │   │   │   ├── 📄 15.htm
│   │   │   │   │   │   │   ├── 📄 16.htm
│   │   │   │   │   │   │   ├── 📄 17.htm
│   │   │   │   │   │   │   ├── 📄 18.htm
│   │   │   │   │   │   │   ├── 📄 19.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 20.htm
│   │   │   │   │   │   │   ├── 📄 21.htm
│   │   │   │   │   │   │   ├── 📄 22.htm
│   │   │   │   │   │   │   ├── 📄 23.htm
│   │   │   │   │   │   │   ├── 📄 24.htm
│   │   │   │   │   │   │   ├── 📄 25.htm
│   │   │   │   │   │   │   ├── 📄 26.htm
│   │   │   │   │   │   │   ├── 📄 27.htm
│   │   │   │   │   │   │   ├── 📄 28.htm
│   │   │   │   │   │   │   ├── 📄 29.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 30.htm
│   │   │   │   │   │   │   ├── 📄 31.htm
│   │   │   │   │   │   │   ├── 📄 32.htm
│   │   │   │   │   │   │   ├── 📄 33.htm
│   │   │   │   │   │   │   ├── 📄 34.htm
│   │   │   │   │   │   │   ├── 📄 35.htm
│   │   │   │   │   │   │   ├── 📄 36.htm
│   │   │   │   │   │   │   ├── 📄 37.htm
│   │   │   │   │   │   │   ├── 📄 38.htm
│   │   │   │   │   │   │   ├── 📄 39.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 📄 8.htm
│   │   │   │   │   │   │   ├── 📄 9.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Flow Summary.htm
│   │   │   │   │   │   ├── 📂 final-Flow Summary.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Timing Analyzer.htm
│   │   │   │   │   │   └── 📂 final-Timing Analyzer.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 10.htm
│   │   │   │   │   │   │   ├── 📄 11.htm
│   │   │   │   │   │   │   ├── 📄 12.htm
│   │   │   │   │   │   │   ├── 📄 13.htm
│   │   │   │   │   │   │   ├── 📄 14.htm
│   │   │   │   │   │   │   ├── 📄 15.htm
│   │   │   │   │   │   │   ├── 📄 16.htm
│   │   │   │   │   │   │   ├── 📄 17.htm
│   │   │   │   │   │   │   ├── 📄 18.htm
│   │   │   │   │   │   │   ├── 📄 19.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 20.htm
│   │   │   │   │   │   │   ├── 📄 21.htm
│   │   │   │   │   │   │   ├── 📄 22.htm
│   │   │   │   │   │   │   ├── 📄 23.htm
│   │   │   │   │   │   │   ├── 📄 24.htm
│   │   │   │   │   │   │   ├── 📄 25.htm
│   │   │   │   │   │   │   ├── 📄 26.htm
│   │   │   │   │   │   │   ├── 📄 27.htm
│   │   │   │   │   │   │   ├── 📄 28.htm
│   │   │   │   │   │   │   ├── 📄 29.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 30.htm
│   │   │   │   │   │   │   ├── 📄 31.htm
│   │   │   │   │   │   │   ├── 📄 32.htm
│   │   │   │   │   │   │   ├── 📄 33.htm
│   │   │   │   │   │   │   ├── 📄 34.htm
│   │   │   │   │   │   │   ├── 📄 35.htm
│   │   │   │   │   │   │   ├── 📄 36.htm
│   │   │   │   │   │   │   ├── 📄 37.htm
│   │   │   │   │   │   │   ├── 📄 38.htm
│   │   │   │   │   │   │   ├── 📄 39.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 40.htm
│   │   │   │   │   │   │   ├── 📄 41.htm
│   │   │   │   │   │   │   ├── 📄 42.htm
│   │   │   │   │   │   │   ├── 📄 43.htm
│   │   │   │   │   │   │   ├── 📄 44.htm
│   │   │   │   │   │   │   ├── 📄 45.htm
│   │   │   │   │   │   │   ├── 📄 46.htm
│   │   │   │   │   │   │   ├── 📄 47.htm
│   │   │   │   │   │   │   ├── 📄 48.htm
│   │   │   │   │   │   │   ├── 📄 49.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 50.htm
│   │   │   │   │   │   │   ├── 📄 51.htm
│   │   │   │   │   │   │   ├── 📄 52.htm
│   │   │   │   │   │   │   ├── 📄 53.htm
│   │   │   │   │   │   │   ├── 📄 54.htm
│   │   │   │   │   │   │   ├── 📄 55.htm
│   │   │   │   │   │   │   ├── 📄 56.htm
│   │   │   │   │   │   │   ├── 📄 57.htm
│   │   │   │   │   │   │   ├── 📄 58.htm
│   │   │   │   │   │   │   ├── 📄 59.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 60.htm
│   │   │   │   │   │   │   ├── 📄 61.htm
│   │   │   │   │   │   │   ├── 📄 62.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 📄 8.htm
│   │   │   │   │   │   │   ├── 📄 9.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   ├── 📂 m_3/
│   │   │   │   │   │   ├── 🖼️ fast_1100mV_0c_model.png
│   │   │   │   │   │   ├── 🖼️ fast_1100mV_85c_model.png
│   │   │   │   │   │   ├── 🖼️ slow_1100mV_0c_model.png
│   │   │   │   │   │   └── 🖼️ slow_1100mV_85c_model.png
│   │   │   │   │   └── 📂 m_8/
│   │   │   │   │   │   ├── 📄 final-Analysis - Synthesis.htm
│   │   │   │   │   │   ├── 📂 final-Analysis - Synthesis.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 10.htm
│   │   │   │   │   │   │   ├── 📄 11.htm
│   │   │   │   │   │   │   ├── 📄 12.htm
│   │   │   │   │   │   │   ├── 📄 13.htm
│   │   │   │   │   │   │   ├── 📄 14.htm
│   │   │   │   │   │   │   ├── 📄 15.htm
│   │   │   │   │   │   │   ├── 📄 16.htm
│   │   │   │   │   │   │   ├── 📄 17.htm
│   │   │   │   │   │   │   ├── 📄 18.htm
│   │   │   │   │   │   │   ├── 📄 19.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 20.htm
│   │   │   │   │   │   │   ├── 📄 21.htm
│   │   │   │   │   │   │   ├── 📄 22.htm
│   │   │   │   │   │   │   ├── 📄 23.htm
│   │   │   │   │   │   │   ├── 📄 24.htm
│   │   │   │   │   │   │   ├── 📄 25.htm
│   │   │   │   │   │   │   ├── 📄 26.htm
│   │   │   │   │   │   │   ├── 📄 27.htm
│   │   │   │   │   │   │   ├── 📄 28.htm
│   │   │   │   │   │   │   ├── 📄 29.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 30.htm
│   │   │   │   │   │   │   ├── 📄 31.htm
│   │   │   │   │   │   │   ├── 📄 32.htm
│   │   │   │   │   │   │   ├── 📄 33.htm
│   │   │   │   │   │   │   ├── 📄 34.htm
│   │   │   │   │   │   │   ├── 📄 35.htm
│   │   │   │   │   │   │   ├── 📄 36.htm
│   │   │   │   │   │   │   ├── 📄 37.htm
│   │   │   │   │   │   │   ├── 📄 38.htm
│   │   │   │   │   │   │   ├── 📄 39.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 40.htm
│   │   │   │   │   │   │   ├── 📄 41.htm
│   │   │   │   │   │   │   ├── 📄 42.htm
│   │   │   │   │   │   │   ├── 📄 43.htm
│   │   │   │   │   │   │   ├── 📄 44.htm
│   │   │   │   │   │   │   ├── 📄 45.htm
│   │   │   │   │   │   │   ├── 📄 46.htm
│   │   │   │   │   │   │   ├── 📄 47.htm
│   │   │   │   │   │   │   ├── 📄 48.htm
│   │   │   │   │   │   │   ├── 📄 49.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 📄 8.htm
│   │   │   │   │   │   │   ├── 📄 9.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Assembler.htm
│   │   │   │   │   │   ├── 📂 final-Assembler.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Resource Usage Summary.htm
│   │   │   │   │   │   ├── 📂 final-Resource Usage Summary.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   ├── 📄 final-Timing Analyzer.htm
│   │   │   │   │   │   ├── 📂 final-Timing Analyzer.htm_files/
│   │   │   │   │   │   │   ├── 📄 1.htm
│   │   │   │   │   │   │   ├── 📄 10.htm
│   │   │   │   │   │   │   ├── 📄 11.htm
│   │   │   │   │   │   │   ├── 📄 12.htm
│   │   │   │   │   │   │   ├── 📄 13.htm
│   │   │   │   │   │   │   ├── 📄 14.htm
│   │   │   │   │   │   │   ├── 📄 15.htm
│   │   │   │   │   │   │   ├── 📄 16.htm
│   │   │   │   │   │   │   ├── 📄 17.htm
│   │   │   │   │   │   │   ├── 📄 18.htm
│   │   │   │   │   │   │   ├── 📄 19.htm
│   │   │   │   │   │   │   ├── 📄 2.htm
│   │   │   │   │   │   │   ├── 📄 20.htm
│   │   │   │   │   │   │   ├── 📄 21.htm
│   │   │   │   │   │   │   ├── 📄 22.htm
│   │   │   │   │   │   │   ├── 📄 23.htm
│   │   │   │   │   │   │   ├── 📄 24.htm
│   │   │   │   │   │   │   ├── 📄 25.htm
│   │   │   │   │   │   │   ├── 📄 26.htm
│   │   │   │   │   │   │   ├── 📄 27.htm
│   │   │   │   │   │   │   ├── 📄 28.htm
│   │   │   │   │   │   │   ├── 📄 29.htm
│   │   │   │   │   │   │   ├── 📄 3.htm
│   │   │   │   │   │   │   ├── 📄 30.htm
│   │   │   │   │   │   │   ├── 📄 31.htm
│   │   │   │   │   │   │   ├── 📄 32.htm
│   │   │   │   │   │   │   ├── 📄 33.htm
│   │   │   │   │   │   │   ├── 📄 34.htm
│   │   │   │   │   │   │   ├── 📄 35.htm
│   │   │   │   │   │   │   ├── 📄 36.htm
│   │   │   │   │   │   │   ├── 📄 37.htm
│   │   │   │   │   │   │   ├── 📄 38.htm
│   │   │   │   │   │   │   ├── 📄 39.htm
│   │   │   │   │   │   │   ├── 📄 4.htm
│   │   │   │   │   │   │   ├── 📄 40.htm
│   │   │   │   │   │   │   ├── 📄 41.htm
│   │   │   │   │   │   │   ├── 📄 42.htm
│   │   │   │   │   │   │   ├── 📄 43.htm
│   │   │   │   │   │   │   ├── 📄 44.htm
│   │   │   │   │   │   │   ├── 📄 45.htm
│   │   │   │   │   │   │   ├── 📄 46.htm
│   │   │   │   │   │   │   ├── 📄 47.htm
│   │   │   │   │   │   │   ├── 📄 48.htm
│   │   │   │   │   │   │   ├── 📄 49.htm
│   │   │   │   │   │   │   ├── 📄 5.htm
│   │   │   │   │   │   │   ├── 📄 50.htm
│   │   │   │   │   │   │   ├── 📄 51.htm
│   │   │   │   │   │   │   ├── 📄 52.htm
│   │   │   │   │   │   │   ├── 📄 53.htm
│   │   │   │   │   │   │   ├── 📄 54.htm
│   │   │   │   │   │   │   ├── 📄 55.htm
│   │   │   │   │   │   │   ├── 📄 56.htm
│   │   │   │   │   │   │   ├── 📄 57.htm
│   │   │   │   │   │   │   ├── 📄 58.htm
│   │   │   │   │   │   │   ├── 📄 59.htm
│   │   │   │   │   │   │   ├── 📄 6.htm
│   │   │   │   │   │   │   ├── 📄 60.htm
│   │   │   │   │   │   │   ├── 📄 61.htm
│   │   │   │   │   │   │   ├── 📄 62.htm
│   │   │   │   │   │   │   ├── 📄 7.htm
│   │   │   │   │   │   │   ├── 📄 8.htm
│   │   │   │   │   │   │   ├── 📄 9.htm
│   │   │   │   │   │   │   ├── 🎨 css/
│   │   │   │   │   │   │   │   ├── 🎨 base.css
│   │   │   │   │   │   │   │   ├── 🖼️ images/
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_0_aaaaaa_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_cccccc_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_75_ffffff_40x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_55_fbf9ee_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_65_ffffff_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_dadada_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_75_e6e6e6_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_95_fef1ec_1x400.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_75_cccccc_1x100.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_222222_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_454545_256x240.png
│   │   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_888888_256x240.png
│   │   │   │   │   │   │   │   │   └── 🖼️ ui-icons_cd0a0a_256x240.png
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui-1.9.2.custom.min.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery-ui.css
│   │   │   │   │   │   │   │   ├── 🎨 jquery.layout-latest.css
│   │   │   │   │   │   │   │   ├── 🎨 normalize.css
│   │   │   │   │   │   │   │   ├── 🎨 override.css
│   │   │   │   │   │   │   │   ├── 🎨 reset.css
│   │   │   │   │   │   │   │   ├── 🎨 server.css
│   │   │   │   │   │   │   │   └── 🎨 sscq_home.css
│   │   │   │   │   │   │   ├── 📂 img/
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_bar_chart.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_closed_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_console_msg_stdout.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_generic_file.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_histogram.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_input_small.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_critical_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_debug.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_error.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_extra_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_info.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_cont_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_tcl_prompt.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_msg_warning.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_new_prj_wiz.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project_48x48.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_opened_folder.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_question_mark.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_report_path.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_compile.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_map.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_online_demo.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_start_screen_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_summary_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_timing_table.png
│   │   │   │   │   │   │   │   ├── 🖼️ afcq_waveform.png
│   │   │   │   │   │   │   │   ├── 🖼️ altera_npo_intel_wht_rgb.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_altera_logo.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_bg.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_buy_sw.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_literature.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_new_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_notifications.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_open_project.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_support.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_training.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_web_vs_sub.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq_whats_new.png
│   │   │   │   │   │   │   │   ├── 🖼️ sscq-intel-white.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_diagonals-thick_90_eeeeee_40x40.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_flat_15_cd0a0a_40x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_100_e4f1fb_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_50_3baae3_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_glass_80_d7ebf9_1x400.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_100_f2f5f7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-hard_70_000000_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_100_deedf7_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-bg_highlight-soft_25_ffef8f_1x100.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2694e8_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_2e83ff_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_3d80b3_256x240.png
│   │   │   │   │   │   │   │   ├── 🖼️ ui-icons_72a7cf_256x240.png
│   │   │   │   │   │   │   │   └── 🖼️ ui-icons_ffffff_256x240.png
│   │   │   │   │   │   │   └── 📂 js/
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-1.7.2.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui-1.9.2.custom.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery-ui.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.layout-latest.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.min.js
│   │   │   │   │   │   │   │   ├── 📜 jquery.periodicalupdater.js
│   │   │   │   │   │   │   │   ├── 📜 sscq_home.js
│   │   │   │   │   │   │   │   └── 📜 sscq_test.js
│   │   │   │   │   │   └── 📄 timing_analysis.xlsx
│   │   │   ├── 📂 v1 _structured_alexnet/
│   │   │   │   ├── 📂 12.5/
│   │   │   │   │   ├── 🖼️ model_sim_12_5.png
│   │   │   │   │   ├── 🖼️ python_output_12_5.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 25/
│   │   │   │   │   ├── 🖼️ model_sim_25.png
│   │   │   │   │   ├── 🖼️ python_output_25.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 37.5/
│   │   │   │   │   ├── 🖼️ model_sim_37_5.png
│   │   │   │   │   ├── 🖼️ python_output_37_5.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 50/
│   │   │   │   │   ├── 🖼️ model_sim_50.png
│   │   │   │   │   ├── 🖼️ python_output_50.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 62.5/
│   │   │   │   │   ├── 🖼️ model_sim_62_5.png
│   │   │   │   │   ├── 🖼️ python_output_62_5.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 75/
│   │   │   │   └── 📂 87.5/
│   │   │   ├── 📂 v1_random/
│   │   │   │   ├── 📂 0/
│   │   │   │   │   ├── 🖼️ model_sim_0.png
│   │   │   │   │   ├── 🖼️ python_output_0.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 10/
│   │   │   │   │   ├── 🖼️ model_sim_10.png
│   │   │   │   │   ├── 🖼️ python_output_10.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 20/
│   │   │   │   │   ├── 🖼️ model_sim_20.png
│   │   │   │   │   ├── 🖼️ python_output_20.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 30/
│   │   │   │   │   ├── 🖼️ model_sim_30.png
│   │   │   │   │   ├── 🖼️ python_output_30.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 40/
│   │   │   │   │   ├── 🖼️ model_sim_40.png
│   │   │   │   │   ├── 🖼️ python_output_40.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 50/
│   │   │   │   │   ├── 🖼️ model_sim_50.png
│   │   │   │   │   ├── 🖼️ python_output_50.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 60/
│   │   │   │   │   ├── 🖼️ model_sim_60.png
│   │   │   │   │   ├── 🖼️ python_output_60.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 70/
│   │   │   │   │   ├── 🖼️ model_sim_70.png
│   │   │   │   │   ├── 🖼️ python_output_70.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 75/
│   │   │   │   │   ├── 🖼️ model_sim_75.png
│   │   │   │   │   ├── 🖼️ python_output_75.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 80/
│   │   │   │   │   ├── 🖼️ model_sim_80.png
│   │   │   │   │   ├── 🖼️ python_output_80.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 85/
│   │   │   │   │   ├── 🖼️ model_sim_85.png
│   │   │   │   │   ├── 🖼️ python_output_85.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 90/
│   │   │   │   │   ├── 🖼️ model_sim_90.png
│   │   │   │   │   ├── 🖼️ python_output_90.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   ├── 📂 95/
│   │   │   │   │   ├── 🖼️ model_sim_95.png
│   │   │   │   │   ├── 🖼️ python_output_95.png
│   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   └── 📄 stripped_weight.mif
│   │   │   │   └── 📄 random_results.xlsx
│   │   │   └── 📂 v2_alexnet/
│   │   │   │   ├── 📂 run_1/
│   │   │   │   │   ├── 📂 tile_0/
│   │   │   │   │   │   ├── 📄 activation_tile_0.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_0.png
│   │   │   │   │   │   └── 📄 weight_tile_0.mif
│   │   │   │   │   ├── 📂 tile_1/
│   │   │   │   │   │   ├── 📄 activation_tile_1.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_1.png
│   │   │   │   │   │   └── 📄 weight_tile_1.mif
│   │   │   │   │   ├── 📂 tile_2/
│   │   │   │   │   │   ├── 📄 activation_tile_2.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_2.png
│   │   │   │   │   │   └── 📄 weight_tile_2.mif
│   │   │   │   │   ├── 📂 tile_3/
│   │   │   │   │   │   ├── 📄 activation_tile_3.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_3.png
│   │   │   │   │   │   └── 📄 weight_tile_3.mif
│   │   │   │   │   ├── 📂 tile_4/
│   │   │   │   │   │   ├── 📄 activation_tile_4.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_4.png
│   │   │   │   │   │   └── 📄 weight_tile_4.mif
│   │   │   │   │   ├── 📂 tile_5/
│   │   │   │   │   │   ├── 📄 activation_tile_5.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_5.png
│   │   │   │   │   │   └── 📄 weight_tile_5.mif
│   │   │   │   │   ├── 📂 tile_6/
│   │   │   │   │   │   ├── 📄 activation_tile_6.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_6.png
│   │   │   │   │   │   └── 📄 weight_tile_6.mif
│   │   │   │   │   └── 📂 tile_7/
│   │   │   │   │   │   ├── 📄 activation_tile_7.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   └── 📄 weight_tile_7.mif
│   │   │   │   ├── 📄 run_1_terminal_output.txt
│   │   │   │   ├── 📂 run_2/
│   │   │   │   │   ├── 📂 tile_0/
│   │   │   │   │   │   ├── 📄 activation_tile_0.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_0.png
│   │   │   │   │   │   └── 📄 weight_tile_0.mif
│   │   │   │   │   ├── 📂 tile_1/
│   │   │   │   │   │   ├── 📄 activation_tile_1.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_1.png
│   │   │   │   │   │   └── 📄 weight_tile_1.mif
│   │   │   │   │   ├── 📂 tile_2/
│   │   │   │   │   │   ├── 📄 activation_tile_2.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_2.png
│   │   │   │   │   │   └── 📄 weight_tile_2.mif
│   │   │   │   │   ├── 📂 tile_3/
│   │   │   │   │   │   ├── 📄 activation_tile_3.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   └── 📄 weight_tile_3.mif
│   │   │   │   │   ├── 📂 tile_4/
│   │   │   │   │   │   ├── 📄 activation_tile_4.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_4.png
│   │   │   │   │   │   └── 📄 weight_tile_4.mif
│   │   │   │   │   ├── 📂 tile_5/
│   │   │   │   │   │   ├── 📄 activation_tile_5.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_5.png
│   │   │   │   │   │   └── 📄 weight_tile_5.mif
│   │   │   │   │   ├── 📂 tile_6/
│   │   │   │   │   │   ├── 📄 activation_tile_6.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_6.png
│   │   │   │   │   │   └── 📄 weight_tile_6.mif
│   │   │   │   │   └── 📂 tile_7/
│   │   │   │   │   │   ├── 📄 activation_tile_7.mif
│   │   │   │   │   │   ├── 📄 stripped_activation.mif
│   │   │   │   │   │   ├── 📄 stripped_weight.mif
│   │   │   │   │   │   ├── 🖼️ tile_7.png
│   │   │   │   │   │   └── 📄 weight_tile_7.mif
│   │   │   │   ├── 📄 run_2_terminal_output.txt
│   │   │   │   ├── 📂 run_3/
│   │   │   │   │   ├── 📂 tile_0/
│   │   │   │   │   ├── 📂 tile_1/
│   │   │   │   │   ├── 📂 tile_2/
│   │   │   │   │   ├── 📂 tile_3/
│   │   │   │   │   ├── 📂 tile_4/
│   │   │   │   │   ├── 📂 tile_5/
│   │   │   │   │   ├── 📂 tile_6/
│   │   │   │   │   └── 📂 tile_7/
│   │   │   │   └── 📄 v2_alexnet_results.xlsx
│   │   └── 📂 work/
│   │   │   ├── 📄 _info
│   │   │   ├── 📂 _temp/
│   │   │   ├── 📄 _vmake
│   │   │   ├── 📂 control_unit/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 behaviour.dat
│   │   │   │   ├── 📄 behaviour.dbs
│   │   │   │   ├── 📄 behaviour.prw
│   │   │   │   └── 📄 behaviour.psm
│   │   │   ├── 📂 custom_types/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 _vhdl.prw
│   │   │   │   └── 📄 _vhdl.psm
│   │   │   ├── 📂 data_rom/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 mock.dat
│   │   │   │   ├── 📄 mock.dbs
│   │   │   │   ├── 📄 mock.prw
│   │   │   │   ├── 📄 mock.psm
│   │   │   │   ├── 📄 syn.dat
│   │   │   │   ├── 📄 syn.dbs
│   │   │   │   ├── 📄 syn.prw
│   │   │   │   └── 📄 syn.psm
│   │   │   ├── 📂 npu_wrapper/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 rtl.dat
│   │   │   │   ├── 📄 rtl.dbs
│   │   │   │   ├── 📄 rtl.prw
│   │   │   │   └── 📄 rtl.psm
│   │   │   ├── 📂 processing_element/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 behaviour.dat
│   │   │   │   ├── 📄 behaviour.dbs
│   │   │   │   ├── 📄 behaviour.prw
│   │   │   │   └── 📄 behaviour.psm
│   │   │   ├── 📂 systolic_array/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 behaviour.dat
│   │   │   │   ├── 📄 behaviour.dbs
│   │   │   │   ├── 📄 behaviour.prw
│   │   │   │   └── 📄 behaviour.psm
│   │   │   ├── 📂 tb_memory/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 sim.dat
│   │   │   │   ├── 📄 sim.dbs
│   │   │   │   ├── 📄 sim.prw
│   │   │   │   └── 📄 sim.psm
│   │   │   ├── 📂 tb_npu_wrapper/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 sim.dat
│   │   │   │   ├── 📄 sim.dbs
│   │   │   │   ├── 📄 sim.prw
│   │   │   │   └── 📄 sim.psm
│   │   │   ├── 📂 tb_row_stripping_sparsity/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   └── 📄 _primary.dbs
│   │   │   ├── 📂 top_level_systolic_array/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 structure.dat
│   │   │   │   ├── 📄 structure.dbs
│   │   │   │   ├── 📄 structure.prw
│   │   │   │   └── 📄 structure.psm
│   │   │   └── 📂 weight_rom/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 mock.dat
│   │   │   │   ├── 📄 mock.dbs
│   │   │   │   ├── 📄 mock.prw
│   │   │   │   ├── 📄 mock.psm
│   │   │   │   ├── 📄 syn.dat
│   │   │   │   ├── 📄 syn.dbs
│   │   │   │   ├── 📄 syn.prw
│   │   │   │   └── 📄 syn.psm
│   ├── 📂 Systolic Array (basic)/
│   │   ├── 📄 array_length_tb.vhd
│   │   ├── 📄 matrix_type.vhd
│   │   ├── 🖼️ Multiplication-of-3-by-3-Matrices-01.png
│   │   ├── 📄 processing_element.vhd
│   │   ├── 🖼️ systolic_array_diagram.jpg
│   │   ├── 📄 systolic_array_tb.vhd
│   │   ├── 📄 systolic_array.vhd
│   │   ├── 📂 work/
│   │   │   └── 📂 _temp/
│   │   └── 🖼️ youtube_ss.png
│   └── 📂 Systolic Array (power-gating)/
│   │   ├── 📄 control_unit.vhd
│   │   ├── 📄 custom_types.vhd
│   │   ├── 📄 pe_output.m
│   │   ├── 📄 processing_element.vhd
│   │   ├── 📄 systolic_array.vhd
│   │   ├── 📄 tb_buffer.vhd
│   │   ├── 📄 tb_systolic_array.vhd
│   │   ├── 📄 tb_top_level_systolic_array.vhd
│   │   ├── 📄 top_level_systolic_array.vhd
│   │   ├── 📄 vsim.wlf
│   │   └── 📂 work/
│   │   │   ├── 📄 _info
│   │   │   ├── 📂 _temp/
│   │   │   ├── 📄 _vmake
│   │   │   ├── 📂 control_unit/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 behaviour.dat
│   │   │   │   ├── 📄 behaviour.dbs
│   │   │   │   ├── 📄 behaviour.prw
│   │   │   │   └── 📄 behaviour.psm
│   │   │   ├── 📂 custom_types/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 _vhdl.prw
│   │   │   │   └── 📄 _vhdl.psm
│   │   │   ├── 📂 processing_element/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 behaviour.dat
│   │   │   │   ├── 📄 behaviour.dbs
│   │   │   │   ├── 📄 behaviour.prw
│   │   │   │   └── 📄 behaviour.psm
│   │   │   ├── 📂 systolic_array/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 behaviour.dat
│   │   │   │   ├── 📄 behaviour.dbs
│   │   │   │   ├── 📄 behaviour.prw
│   │   │   │   └── 📄 behaviour.psm
│   │   │   ├── 📂 tb_top_level_systolic_array/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 sim.dat
│   │   │   │   ├── 📄 sim.dbs
│   │   │   │   ├── 📄 sim.prw
│   │   │   │   └── 📄 sim.psm
│   │   │   └── 📂 top_level_systolic_array/
│   │   │   │   ├── 📄 _primary.dat
│   │   │   │   ├── 📄 _primary.dbs
│   │   │   │   ├── 📄 structure.dat
│   │   │   │   ├── 📄 structure.dbs
│   │   │   │   ├── 📄 structure.prw
│   │   │   │   └── 📄 structure.psm
├── 📂 figures/
│   ├── 🖼️ baseline_systolic_array_dataflow.jpg
│   ├── 🖼️ baseline_systolic_array.jpg
│   ├── 🖼️ cat.jpg
│   ├── 🖼️ date.png
│   ├── 🖼️ fsm_npu.jpg
│   ├── 🖼️ MAC.jpg
│   ├── 🖼️ rtl.jpg
│   ├── 🖼️ signature.png
│   ├── 🖼️ system_diagram.jpg
│   ├── 🖼️ tb_fsm.jpg
│   └── 🖼️ UoA-Logo-Primary-RGB-Large.png
├── 📂 general/
│   ├── 📄 Meeting Minutes.docx
│   ├── 📄 Project_43_Seminar_Slides.pptx
│   └── 📄 Resource_list.xlsx
├── 📂 Kristelle/
│   ├── 🖼️ date.png
│   ├── 📄 final_report.bib
│   ├── 📕 P4P Literature Review Mind Map.pdf
│   ├── 📕 P4P_Kristelle_Sampang_Mid_Year_Report.pdf
│   ├── 📕 P4P_Kristelle_Sampang_Project_Scope_Objectives_Literature_Review_Report_v2.pdf
│   ├── 📄 P4P_Kristelle_Sampang_Project_Scope_Objectives_Literature_Review_Report.docx
│   ├── 📕 P4P_Kristelle_Sampang_Project_Scope_Objectives_Literature_Review_Report.pdf
│   ├── 📕 Project-Based Risk Assessment_2025-signed.pdf
│   └── 🖼️ signature.png
├── 📂 Pratham/
├── 📄 projectStructure.exclusions
├── 📂 results/
│   ├── 📂 baseline_modelsim/
│   │   ├── 🖼️ 1x1_matlab.png
│   │   ├── 🖼️ 1x1.png
│   │   ├── 🖼️ 1x8_matlab.png
│   │   ├── 🖼️ 1x8.png
│   │   ├── 🖼️ 2x2_matlab.png
│   │   ├── 🖼️ 2x2.png
│   │   ├── 🖼️ 2x8_matlab.png
│   │   ├── 🖼️ 2x8.png
│   │   ├── 🖼️ 3x3_matlab.png
│   │   ├── 🖼️ 3x3.png
│   │   ├── 🖼️ 3x8_matlab.png
│   │   ├── 🖼️ 3x8.png
│   │   ├── 🖼️ 4x4_matlab.png
│   │   ├── 🖼️ 4x4.png
│   │   ├── 🖼️ 4x8_matlab.png
│   │   ├── 🖼️ 4x8.png
│   │   ├── 🖼️ 5x5_matlab.png
│   │   ├── 🖼️ 5x5.png
│   │   ├── 🖼️ 5x8_matlab.png
│   │   ├── 🖼️ 5x8.png
│   │   ├── 🖼️ 6x6_matlab.png
│   │   ├── 🖼️ 6x6.png
│   │   ├── 🖼️ 6x8_matlab.png
│   │   ├── 🖼️ 6x8.png
│   │   ├── 🖼️ 7x7_matlab.png
│   │   ├── 🖼️ 7x7.png
│   │   ├── 🖼️ 7x8_matlab.png
│   │   ├── 🖼️ 7x8.png
│   │   ├── 🖼️ 8x8_matlab.png
│   │   └── 🖼️ 8x8.png
│   ├── 🖼️ Effect of M value on the number of Active PEs in the Systolic Array and Latency v2.png
│   ├── 🖼️ Effect of M value on the number of Active PEs in the Systolic Array and Latency.png
│   ├── 🖼️ fmax.png
│   ├── 🖼️ Impact of Stripping Algorithm on Clock Cycles when Applied on Random Sparsity Graph.png
│   ├── 🖼️ Impact of Stripping Algorithm on Clock Cycles when Applied on Random Sparsity.png
│   ├── 🖼️ Impact on Latency as Row Stripping Increases in a 8x8 Activation Graph.png
│   ├── 🖼️ Impact on Latency as Row Stripping Increases in a 8x8 Activation.png
│   ├── 🖼️ Latency of Baseline Systolic Array on Different NxN sizes.png
│   ├── 🖼️ Latency of Baseline Systolic Array on Varying Input Matrix Size.png
│   ├── 📂 optimised/
│   │   ├── 🖼️ m0_py.png
│   │   ├── 🖼️ m0.png
│   │   ├── 🖼️ m1_py.png
│   │   ├── 🖼️ m1.png
│   │   ├── 🖼️ m2_py.png
│   │   ├── 🖼️ m2.png
│   │   ├── 🖼️ m3_py.png
│   │   ├── 🖼️ m3.png
│   │   ├── 🖼️ m4_py.png
│   │   ├── 🖼️ m4.png
│   │   ├── 🖼️ m5_py.png
│   │   ├── 🖼️ m5.png
│   │   ├── 🖼️ m6_py.png
│   │   ├── 🖼️ m6.png
│   │   ├── 🖼️ m7_py.png
│   │   ├── 🖼️ m7.png
│   │   ├── 🖼️ m8_py.png
│   │   └── 🖼️ m8.png
│   └── 🖼️ Performance Comparison - Average Clock Cycles per Tile of an AlexNet Convolutional Layer.png
└── 📂 work/
│   ├── 📂 _temp/
│   ├── 📂 control_unit/
│   ├── 📂 custom_types/
│   ├── 📂 databuffer/
│   ├── 📂 processing_element/
│   ├── 📂 systolic_array/
│   ├── 📂 tb_buffer/
│   ├── 📂 tb_top_level_systolic_array/
│   ├── 📂 top_level_systolic_array/
│   └── 📂 weightbuffer/
```

## 📖 Legend

### File Types

- 🚫 DevOps: Git ignore
- 📄 Other: Other files
- ⚙️ Config: JSON files
- 🖼️ Assets: JPEG images
- ⚙️ Config: XML files
- 📄 Docs: Text files
- 🌐 Web: HTML files
- 🎨 Styles: Stylesheets
- 🖼️ Assets: PNG images
- 📜 JavaScript: JavaScript files
- 📕 Docs: PDF files

### Importance Levels

- 🔴 Critical: Essential project files
- 🟡 High: Important configuration files
- 🔵 Medium: Helpful but not essential files
