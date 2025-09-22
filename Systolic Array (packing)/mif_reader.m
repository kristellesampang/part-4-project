% This is a script to call our MIF reader function

% 1. Define the path to your .mif file
mif_filepath = 'C:/Users/iamkr/Documents/part-4-project/Systolic Array (packing)/mif/basic_packing_algo.mif';

% 2. Call the function and store the returned array in a variable
my_data_array = mif_to_1d_array_matlab(mif_filepath);

% for every 8 elements, create a new row to create a 2D array
my_data_array_2d = reshape(my_data_array, 8, []).';

% 3. Display the result
disp('--- 1D Array loaded from .mif file ---');
disp(my_data_array);