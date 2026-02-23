cmake_minimum_required(VERSION 3.14)

add_subdirectory(../../bsp bsp)

include(../../bsp/toolchain.cmake)

project(SummerResearchNonDPR)

enable_language(ASM)
enable_language(C)
enable_language(CXX)

add_executable(SummerResearchNonDPR.elf)

target_sources(SummerResearchNonDPR.elf
    PRIVATE
        src/main.c
)

target_include_directories(SummerResearchNonDPR.elf
    PRIVATE
    PUBLIC
)

target_link_libraries(SummerResearchNonDPR.elf
    PRIVATE
        -T "${BspLinkerScript}" --specs=nosys.specs
        "${ExtraArchiveLibraries}"
        -Wl,--start-group "${BspLibraryName}" -lc -lstdc++ -lgcc -lm -Wl,--end-group
)

# Create objdump from ELF.
set(objdump SummerResearchNonDPR.elf.objdump)
add_custom_command(
    OUTPUT "${objdump}"
    DEPENDS SummerResearchNonDPR.elf
    COMMAND "${ToolchainObjdump}" "${ToolchainObjdumpFlags}" SummerResearchNonDPR.elf >
            "${objdump}"
    COMMENT "Creating ${objdump}."
    VERBATIM
)
add_custom_target(create-objdump ALL DEPENDS "${objdump}")

# Report space free for stack + heap. Note that the file below is never created
# so the report is always output on build.
set(stack_report_file SummerResearchNonDPR.elf.stack_report)
add_custom_command(
    OUTPUT "${stack_report_file}"
    DEPENDS SummerResearchNonDPR.elf
    COMMAND niosv-stack-report -p "${ToolchainPrefix}" SummerResearchNonDPR.elf
    COMMENT "Reporting memory available for stack + heap in SummerResearchNonDPR.elf."
    VERBATIM
)
add_custom_target(niosv-stack-report ALL DEPENDS "${stack_report_file}")

# Generate HEX file(s) from SummerResearchNonDPR.elf using elf2hex tool.
# Note : If ECC Full is enabled, width of 39 is set for NiosV TCM. Otherwise, 32.
add_custom_command(
    OUTPUT "data_mem.hex"
    DEPENDS SummerResearchNonDPR.elf
    COMMAND elf2hex SummerResearchNonDPR.elf -o data_mem.hex -b 0x00031000 -w 32 -e 0x00031FFF -r 4
    COMMENT "Creating data_mem.hex."
    VERBATIM
)
add_custom_command(
    OUTPUT "onchip_memory2_0.hex"
    DEPENDS SummerResearchNonDPR.elf
    COMMAND elf2hex SummerResearchNonDPR.elf -o onchip_memory2_0.hex -b 0x00000000 -w 32 -e 0x0000FFFF -r 4
    COMMENT "Creating onchip_memory2_0.hex."
    VERBATIM
)
add_custom_command(
    OUTPUT "out_mem.hex"
    DEPENDS SummerResearchNonDPR.elf
    COMMAND elf2hex SummerResearchNonDPR.elf -o out_mem.hex -b 0x00033000 -w 32 -e 0x00033FFF -r 4
    COMMENT "Creating out_mem.hex."
    VERBATIM
)
add_custom_command(
    OUTPUT "weight_mem.hex"
    DEPENDS SummerResearchNonDPR.elf
    COMMAND elf2hex SummerResearchNonDPR.elf -o weight_mem.hex -b 0x00032000 -w 32 -e 0x00032FFF -r 4
    COMMENT "Creating weight_mem.hex."
    VERBATIM
)
add_custom_target(create-hex ALL DEPENDS "onchip_memory2_0.hex" "data_mem.hex" "weight_mem.hex" "out_mem.hex")
