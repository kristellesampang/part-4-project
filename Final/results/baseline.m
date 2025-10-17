% MATLAB script to verify VHDL systolic array matrix multiplication.
% This script generates matrices A and B using the same logic as the
% VHDL testbench, computes C = A * B, and displays the results.

% Clears the command window, workspace variables, and closes all figures.
clc;
clear;
close all;

% The VHDL testbench loops through sizes from 1 to 8.
for size_val = 1:8
    
    % --- Matrix Generation ---
    % This section mirrors the VHDL process for creating matrices A and B.
    
    % Initialize matrices A and B with zeros.
    % In MATLAB, indexing is 1-based, unlike VHDL's 0-based.
    A = zeros(size_val, size_val);
    B = zeros(size_val, size_val);
    
    % Initialize starting values for matrix elements, same as in VHDL.
    val_a = 1;
    val_b = 10;
    
    % Populate matrices A and B row by row.
    for i = 1:size_val      % Iterates through rows
        for j = 1:size_val  % Iterates through columns
            A(i, j) = val_a;
            B(i, j) = val_b;
            
            % Increment values for the next element.
            val_a = val_a + 1;
            val_b = val_b + 1;
        end
    end
    
    % --- Matrix Multiplication ---
    C = A * B;
    
    % --- Display Results ---
    % This section mimics the 'report' statements in the VHDL testbench.
    fprintf('=== Results for size %dx%d ===\n', size_val, size_val);
    
    disp('Matrix A:');
    disp(A);
    
    disp('Matrix B:');
    disp(B);
    
    disp('Result C = A * B:');
    disp(C);
    
    fprintf('\n----------------------------------------\n\n');
end

fprintf('Simulation completed successfully\n');