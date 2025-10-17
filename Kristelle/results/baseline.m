% MATLAB script to model a VHDL testbench where matrix B is always 8x8.
% It computes C = A * B_sub, where B_sub is a submatrix of B.

% Clears the command window, workspace variables, and closes all figures.
clc;
clear;
close all;

% Define the maximum dimension N, which is 8.
N = 8;

% The main loop iterates through 'size' from 1 to 8.
for size_val = 1:8
    
    % --- Matrix Generation ---
    % Reset counters for each new size, just like in the VHDL testbench.
    val_a = 1;
    val_b = 10;
    
    % 1. Generate Matrix A (size_val x 8)
    % The number of rows depends on the current 'size_val'.
    A = zeros(size_val, N);
    for i = 1:size_val
        for j = 1:N
            A(i, j) = val_a;
            val_a = val_a + 1;
        end
    end
    
    % 2. Generate Matrix B (8 x 8)
    % This matrix is always fully populated, independent of 'size_val'.
    B = zeros(N, N);
    for i = 1:N
        for j = 1:N
            B(i, j) = val_b;
            val_b = val_b + 1;
        end
    end
    
    % --- Matrix Multiplication ---
    % The VHDL report loop expects a 'size_val x size_val' result matrix.
    % To achieve this, we multiply A (size_val x 8) by a submatrix of B.
    % We take all 8 rows but only the first 'size_val' columns of B.
    % C(size x size) = A(size x 8) * B(8 x size)
    
    B_sub = B(:, 1:size_val);
    C = A * B_sub;
    
    % --- Display Results ---
    fprintf('=== Results for size %d ===\n', size_val);
    
    fprintf('Matrix A (%d x %d):\n', size_val, N);
    disp(A);
    
    fprintf('Matrix B (%d x %d) - Full Matrix:\n', N, N);
    disp(B);
    
    fprintf('Result C = A * B(:, 1:%d) -> (%d x %d):\n', size_val, size_val, size_val);
    disp(C);
    
    fprintf('\n----------------------------------------\n\n');
end

fprintf('Simulation completed successfully\n');