% Data Matrix (A)
% A = [0 1 2;
%      3 4 5;
%      6 7 8];


A = reshape(1:64, 8, 8)';


% Weight Matrix (B)
% B = [8 7 6;
%      5 4 3;
%      2 1 0];
B = reshape(10:73, 8, 8)';


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
fprintf('\nDetailed view:\n');
for i = 1:8
    for j = 1:8
        fprintf('PE(%d,%d) = %d\n', i-1, j-1, C(i,j));
    end
end

