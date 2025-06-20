% Copyright........
%%%% coded by Abhinav singh (PhD 4th year, EE)
% Add paths for required directories
%addpath('D:\Shaswata\NDT data\NDTcode');
%addpath('D:\Shaswata\NDT data');
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
load('D:\TUFFC Paper\pDAS_Beamforming\FMCData\Bloc_FMC_VPO.mat');
% load('D:\TUFFC Paper\pDAS_Beamforming\FMCData\Bloc_FMC_HPO.mat');
RF_Data=Data.Acquisition.Acq1.DATA;
[r,~]=size(RF_Data);
BeamformX = [-50:.1:50]*1000; %%VPO
BeamformZ = [20:.1:170]*1000;%%VPO
% BeamformX = [-40:.1:40]*1000;%% change this for HPO
% BeamformZ = [10:.1:70]*1000; %% CHANGE this for HPO
%RF_Data = Data.Acquisition.Acq1.DATA;
%[r, ~] = size(RF_Data);
% Define beamforming grid
% Initialize temporary variables
tem_pcf(length(BeamformX), length(BeamformZ)) = 0;
temp(length(BeamformX), length(BeamformZ)) = 0;
% Loop configuration variables
Co = 1;  %Element configuration %%%% 1>>64,2>>32, 4>>16, 8>>8, 16>>4, 32>>2
p =4;
l = 1:Co:64;
loc = linspace(-19.53, 19.53, 64) .* 1e-3;
loc = loc(1:Co:64);
element_Pos_Array_um_X = loc .* 1e6;
RF_Start_Time = 0;
speed_Of_Sound_umps = 6354 * 10^6;
% Processing loop
for j = 1:length(l)
    disp(['Processing: ', num2str(j)]);
    rf_Data = (RF_Data(:, 64 * l(j) - 63:64 * l(j)));
    rf_Data = rf_Data(:, 1:Co:64);
    Single_element_loc=[element_Pos_Array_um_X(j),0];
    % Beamforming
    [I] = PDAS(rf_Data, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time, fs, BeamformX, BeamformZ, Single_element_loc, p);
    %[I] = SigncoherenceNDT(rf_Data, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time,fs,BeamformX, BeamformZ,Single_element_loc,p);
    temp = abs(hilbert(I));
    tem_pcf = tem_pcf + temp;
    % Visualization (Existing)
    FrameData = abs(hilbert(I));
    tempx = (FrameData ./ (max(FrameData(:))));
    figure(2);
    imagesc(BeamformX*1e-3,BeamformZ*1e-3,(tempx)'); axis equal; axis tight; colormap('hot'); colorbar('Direction','normal'); %caxis([-40 0]);
    C=colorbar;
    C.Label.String='Amplitude(in dB)';
    set(C.Label,'Position',[5,-10.668874227528,0]);
    set(C.Label,'FontSize',22)
    C.Label.Rotation=270;
    %C.Label.Direction='reverse';
    xlabel('Lateral Location (mm)');
    ylabel('Range Location (mm)');
    title('TFM using DAS');
    set(gca,'fontsize', 20);
%cd('D:\Shaswata\NDT data\Sparseimages\HPO\8 elements');
%cd('D:\Beamforming\Literature_Review\NDT_Coding\New_abstract_updated\White_background_VPO\64_elements\P4');
end
I=((tem_pcf'));
%%
%load('64VPODAS1.mat')
%I=I;
%temp=abs(hilbert(I));
tempx=(I./(max(I(:))));
%tempfilt=bandpass(tempx,[1.85 2.15],fs);
figure(3);
imagesc(BeamformX*1e-3,BeamformZ*1e-3,(tempx)); axis equal; axis tight; colormap('hot'); colorbar('Direction','normal'); caxis([0 1])
C=colorbar;
C.Label.String='Amplitude(in dB)';
set(C.Label,'Position',[5,-10.668874227528,0]);
set(C.Label,'FontSize',22)
C.Label.Rotation=270;
%C.Label.Direction='reverse';
xlabel('Lateral Location (mm)');
ylabel('Range Location (mm)');
title(' TFM using DAS');
set(gca,'fontsize', 20);
figure(4);
imagesc(BeamformX * 1e-3, BeamformZ * 1e-3, tempx);
axis equal; 
axis tight;
colorbar('Direction', 'normal'); 
caxis([0 1]);
% Apply custom colormap
% Normalize the data for colormap scaling
%tempx = (I - min(I(:))) / (max(I(:)) - min(I(:)));
% Define a custom colormap with a faded transition from white to blue to green to yellow to red
custom_colormap = [
    1 1 1;    % White
    0 0 1;    % Blue
    0 1 0;    % Green
    1 1 0;    % Yellow
    1 0 0     % Red
];
% Create a colormap with interpolated colors
n = 64;  % Number of colormap entries
colormap_entries = size(custom_colormap, 1);
interp_colormap = interp1(1:colormap_entries, custom_colormap, linspace(1, colormap_entries, n));
% Apply the custom colormap
colormap(interp_colormap);
% Colorbar settings
C = colorbar;
C.Label.String = 'Amplitude (in dB)';
set(C.Label, 'Position', [5, -10.668874227528, 0]);
set(C.Label, 'FontSize', 22);
C.Label.Rotation = 270;

% Axes labels and title
xlabel('Lateral Location (mm)');
ylabel('Range Location (mm)');
title('TFM using DAS with Custom Colormap');
set(gca, 'fontsize', 20);

% % Save the figure and data for figure(4)
% % temp_Image = ['PthDAS', num2str(p) '_Colored.fig'];
% % saveas(gca, temp_Image);
% % save(['TFMmat', num2str(var), num2str(p), '_Colored'], 'I');
% temp_Image = ['16VPODAS', num2str(p),'.fig'];
% saveas(gca, temp_Image);
% save(['16VPODAS', num2str(p)], 'I')
toc