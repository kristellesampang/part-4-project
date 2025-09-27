% This is a script to call our MIF reader function and pack the matrix
clear;
clc;

% 1. Define the path to your .mif file
mif_filepath = 'C:/Users/iamkr/Documents/part-4-project/Systolic Array (packing)/mif/basic_packing_algo.mif';
N = 8; % Define the matrix size

% 2. Call the function to get the 1D array
my_data_array_1d = mif_to_1d_array_matlab(mif_filepath);

% 3. Reshape the 1D array into a 2D matrix
my_data_array_2d = reshape(my_data_array_1d, [N, N])';

disp('Original 8x8 Array:');
disp(my_data_array_2d);

% --- Corrected Column Packing Algorithm ---
% Use a cell array to store the packed groups
packed_groups = {};

% Iterate through each column of the original 2D matrix
for i = 1:size(my_data_array_2d, 2)
    
    candidate_col = my_data_array_2d(:, i);
    
    % Skip columns that are all zero
    if ~any(candidate_col)
        continue;
    end
    
    was_packed = false;
    
    % Try to pack the column into an existing group
    for g = 1:length(packed_groups)
        % Correct collision check: true if any row has a non-zero in both
        collision = any(packed_groups{g} > 0 & candidate_col > 0);
        
        if ~collision
            % No collision, so merge by adding the column to the group
            packed_groups{g} = packed_groups{g} + candidate_col;
            was_packed = true;
            break; % Move to the next candidate column
        end
    end
    
    % If it couldn't be packed into any existing group, create a new one
    if ~was_packed
        packed_groups{end+1} = candidate_col;
    end
end

% --- Reassemble the packed groups into a single matrix for display ---
if ~isempty(packed_groups)
    num_packed_cols = length(packed_groups);
    packed_array = zeros(size(my_data_array_2d, 1), num_packed_cols);
    for i = 1:num_packed_cols
        packed_array(:, i) = packed_groups{i};
    end
else
    packed_array = []; % Handle case where matrix is all zeros
end


% --- Display Results ---
disp('Packed Array:');
disp(packed_array);