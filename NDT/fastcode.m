% Copyright........
%%%% coded by Abhinav singh (PhD 4th year, EE)
% Add paths for required directories

clc; % Clear the command window
close all; % Close all figures
clearvars; % Clear variables
% Initialize variables
tic
no_Of_Elements = 64;
fs = 62.5e6;
c = 6354;
% Load data
% load('Bloc_FMC_VPO.mat');
load('Bloc_FMC_HPO.mat');
RF_Data=Data.Acquisition.Acq1.DATA;
[r,~]=size(RF_Data);
% BeamformX = [-50:.1:50]*1000; %% VPO
% BeamformZ = [20:.1:170]*1000; %% VPO
BeamformX = [-40:.1:40]*1000;%% change this for HPO
BeamformZ = [10:.1:70]*1000; %% CHANGE this for HPO
%RF_Data = Data.Acquisition.Acq1.DATA;
%[r, ~] = size(RF_Data);
% Define beamforming grid
% Initialize temporary variables
tem_pcf(length(BeamformX), length(BeamformZ)) = 0;
temp(length(BeamformX), length(BeamformZ)) = 0;
% Loop configuration variables
Co = 1;  %Element configuration %%%% 1>>64,2>>32, 4>>16, 8>>8, 16>>4, 32>>2
p = 4;
l = 1:Co:64;
loc = linspace(-19.53, 19.53, 64) .* 1e-3;
loc = loc(1:Co:64);
element_Pos_Array_um_X = loc .* 1e6;
RF_Start_Time = 0;
speed_Of_Sound_umps = 6354 * 10^6;
%Processing loop
for j = 1:length(l)
    disp(['Processing: ', num2str(j)]);

    rf_Data = RF_Data(:, 64 * l(j) - 63 : 64 * l(j));
    rf_Data = rf_Data(:, 1:Co:64);
    Single_element_loc = [element_Pos_Array_um_X(j), 0];

    % Beamforming
    [I] = pthcoherenceNDT(rf_Data, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time, fs, BeamformX, BeamformZ, Single_element_loc, p);

    temp = abs(hilbert(I));
    tem_pcf = tem_pcf + temp;
end
% tem_pcf = comb( ...
%     single(RF_Data), ...
%     single(element_Pos_Array_um_X), ...
%     single(l), ...
%     single(Co), ...
%     single(speed_Of_Sound_umps), ...
%     single(RF_Start_Time), ...
%     single(fs), ...
%     single(BeamformX), ...
%     single(BeamformZ), ...
%     single(p));
%     tem_pcf = abs(hilbert(tem_pcf));

I = tem_pcf';
tempx = I ./ max(I(:));

% Final visualization with custom colormap
figure;
imagesc(BeamformX * 1e-3, BeamformZ * 1e-3, tempx);
axis equal;
axis tight;
colorbar('Direction', 'normal');
caxis([0 1]);

% Define and apply custom colormap
custom_colormap = [
    1 1 1;    % White
    0 0 1;    % Blue
    0 1 0;    % Green
    1 1 0;    % Yellow
    1 0 0     % Red
];
n = 64;
interp_colormap = interp1(1:size(custom_colormap, 1), custom_colormap, linspace(1, size(custom_colormap, 1), n));
colormap(interp_colormap);

% Colorbar and label settings
C = colorbar;
C.Label.String = 'Amplitude (in dB)';
set(C.Label, 'Position', [5, -10.668874227528, 0]);
set(C.Label, 'FontSize', 22);
C.Label.Rotation = 270;

% Axes labels and title
xlabel('Lateral Location (mm)');
ylabel('Range Location (mm)');
title('TFM using DAS with Custom Colormap');
set(gca, 'FontSize', 20);

toc
