% Data Matrix (A)
A = [ 1  2  3  4  5  6  7  8;
     9 10 11 12 13 14 15 16;
     17 18 19 20 21 22 23 24;
     25 26 27 28 29 30 31 32;
     33 34 35 36 37 38 39 40;
     41 42 43 44 45 46 47 48;
     49 50 51 52 53 54 55 56;
     57 58 59 60 61 62 63 64];

% Weight Matrix (B)
B = [ 8  7  6  5  4  3  2  1;
     16 15 14 13 12 11 10  9;
     24 23 22 21 20 19 18 17;
     32 31 30 29 28 27 26 25;
     40 39 38 37 36 35 34 33;
     48 47 46 45 44 43 42 41;
     56 55 54 53 52 51 50 49;
     64 63 62 61 60 59 58 57];

C = A * B;          

disp('Accumulated value of each PE (row i, column j):');
disp(C);

% Optional: print with PE labels
%fprintf('\nDetailed view:\n');
%for i = 1:3
%    for j = 1:3
%        fprintf('PE(%d,%d) = %d\n', i-1, j-1, C(i,j));
%    end
%end
