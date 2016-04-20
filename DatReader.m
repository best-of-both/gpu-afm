%DatReader.m
%
% This m-file loads the DAT file created by the eigenvalue solver 
% Linux code: http://developer.download.nvidia.com/compute/cuda/2_3/sdk/projects/eigenvalues.tar.gz

% Prepare our space
close all
format compact
format long
DEBUG=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  REad in data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FILENAME = 'reference.dat';
fileID = fopen(FILENAME);
C1 = textscan(fileID,'%s%f%s%f');   %read the first line

im_x = C1{2};                      %get the x dimension 
num_eigs = size(im_x,1);
%im_y = C1{4}                      %get the x dimension 
yvals = zeros(1,num_eigs);
%C2 = textscan(fileID,'%f%f%f%f%f%f%*[^\n]', 'CollectOutput',1);

fclose(fileID);


if DEBUG
    fprintf('This file contains %d eigenvalues\n', num_eigs)
end % end if 

%%%%%%%%%%%%%%%%%%%%%%%5
%% Plot values
%%%%%%%%%%%%%%%%%%%%%%%%
plot(im_x,yvals, 'r*')
title('Eigenvalues of input matrix')
xlabel('Eigen spectrum')
