function data_array = mif_to_1d_array_matlab(filename)
% Reads a Quartus Memory Initialization File (.mif) and converts it
% into a 1D MATLAB array (vector).

% Open the file for reading
fileID = fopen('C:/Users/iamkr/Documents/part-4-project/Systolic Array (packing)/mif/basic_packing_algo.mif', 'r');
if fileID == -1
    error('Error: The file ''%s'' was not found.', filename);
end

data_array = [];
in_content_section = false;

% Read the file line by line
tline = fgetl(fileID);
while ischar(tline)
    % Start reading only after the 'CONTENT BEGIN' line
    if contains(tline, 'CONTENT BEGIN')
        in_content_section = true;
        tline = fgetl(fileID); % Move to the next line
        continue;
    end
    
    if contains(tline, 'END;')
        break;
    end
    
    if in_content_section
        % Use a regular expression to find "address : value;"
        tokens = regexp(tline, '\d+\s*:\s*([0-9a-fA-F]+);', 'tokens');
        if ~isempty(tokens)
            hex_value = tokens{1}{1};
            % Convert hex value to a decimal number and append
            data_array(end+1) = hex2dec(hex_value);
        end
    end
    
    tline = fgetl(fileID); % Read the next line
end
end