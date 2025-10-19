clear 
clc

N = 8;
count = 0;
mif_filepath = 'C:/Users/iamkr/Documents/part-4-project/Systolic Array (packing)/mif/basic_packing_algo.mif';

my_1d_array = mif_to_1d_array_matlab(mif_filepath);
unpacked_array = reshape(my_1d_array, [N, N])';
disp('Original 8x8 Array:');
disp(unpacked_array);



% count the amount of non-zero elements in the matrix 
for i = 1:N
    for j = 1:N
        if unpacked_array(i,j) ~= 0
            count = count + 1;
        end
    end
end

disp(['Number of non-zero elements: ', num2str(count)]);


% Find the most the smallest NxN array it can fit into
min_size = ceil(sqrt(count));
disp(['Minimum size of square array to fit all non-zero elements: ', num2str(min_size)]);

% Reorganise the matrix into the min_size x min_size matrix, keeping track the original position and its new position using cell arrays
reorganized_array = zeros(min_size, min_size);
original_positions = cell(min_size, min_size);
new_positions = cell(min_size, min_size);
current_row = 1;
current_col = 1;
for i = 1:N
    for j = 1:N
        if unpacked_array(i,j) ~= 0
            reorganized_array(current_row, current_col) = unpacked_array(i,j);
            original_positions{current_row, current_col} = [i, j];
            new_positions{current_row, current_col} = [current_row, current_col];
            current_col = current_col + 1;
            if current_col > min_size
                current_col = 1;
                current_row = current_row + 1;
            end
        end
    end
end
disp('Reorganized Array:');
disp(reorganized_array);
disp('Original Positions of Non-Zero Elements:');
disp(original_positions);
disp('New Positions of Non-Zero Elements:');
disp(new_positions);
