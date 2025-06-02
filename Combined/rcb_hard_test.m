%% Clear workspace and add folder with data
close all;
clearvars;

% Hardcoded path to the data folder
pn = './';
addpath(pn);

% Load the setup file directly
setupFile = fullfile(pn, 'SetUpC5_2v_ChirpPCI_AP.mat');
if exist(setupFile, 'file')
    load(setupFile);
else
    error('Setup file not found at specified path.');
end

disp("Hello")

%% Select a dataset
% Look for a file starting with 'PD' in the folder
pp = dir(fullfile(pn, 'PD*.mat'));
if isempty(pp)
    error('No PD dataset found in the specified folder.');
end

% Pick the first matching PD file
datasetFile = fullfile(pn, pp(1).name);

% Load the dataset
load(datasetFile);  % Expects variable RData in the file

%% Use only the first frame for processing
numSamples = min(size(RData,1), 1472);
rfFrame = double(RData(1:numSamples, 1:128, 1));

%% Define imaging parameters
center_frequency = 3.68e6;
wavelength = (1540e3 / center_frequency);
Aperture = Trans.ElementPos(:, [1 3])';
Aperture = Aperture * wavelength;
element_Pos_Array_um_X = Aperture * 1e3;

sampling_Freq = 4 * center_frequency;
image_Range_X_um = linspace(-30.0901, 30.1707, 66) * 1e3;
image_Range_Z_um = linspace(19.91, 77.6644, 125) * 1e3;
speed_Of_Sound_umps = 1540e6;
RF_Start_Time = 0;

%% Run PCI imaging
arr_rcb_single = [];

for i = 1:20
    numElements = size(element_Pos_Array_um_X, 2);
    [II, delay] = withforwrapper_rcb_20_single(rfFrame, element_Pos_Array_um_X, ...
        speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, ...
        image_Range_X_um, image_Range_Z_um, 3, numElements);

    arr_rcb_single{end+1} = delay;  % Use cell array to store if delay is not a scalar
end

disp('All delay values after 20 iterations for RCB single: ');
disp(arr_rcb_single);
arr_rcb_double = [];

for i = 1:20
    numElements = size(element_Pos_Array_um_X, 2);
    [II, delay] = withforwrapper_rcb_20_double(rfFrame, element_Pos_Array_um_X, ...
        speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, ...
        image_Range_X_um, image_Range_Z_um, 3, numElements);
    
    arr_rcb_double{end+1} = delay;  % Use cell array to store if delay is not a scalar
end

disp('All delay values after 20 iterations for RCB Double: ');
disp(arr_rcb_double);

range_frq = [2e6, 5e6]; 
arr_0_single= [];

for i = 1:20
    numElements = size(element_Pos_Array_um_X, 2);
     % Lower and upper bounds only
    check = 0;

[II,delay] = PCIfreqRBCcuda_single(rfFrame, element_Pos_Array_um_X, ...
    speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, ...
    image_Range_X_um, image_Range_Z_um, 3, range_frq, check);
    
    arr_0_single{end+1} = delay;  % Use cell array to store if delay is not a scalar
end

disp('All delay values after 20 iterations for PCI single: ');
disp(arr_0_single);
arr_0_double= [];

for i = 1:20
    numElements = size(element_Pos_Array_um_X, 2);
    check = 0;

[II,delay] = PCIfreqRBCcuda_double(rfFrame, element_Pos_Array_um_X, ...
    speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, ...
    image_Range_X_um, image_Range_Z_um, 3, range_frq, check);
    
    
    arr_0_double{end+1} = delay;  % Use cell array to store if delay is not a scalar
end

disp('All delay values after 20 iterations for PCI Double: ');
disp(arr_0_double);



arr_1_single= [];

for i = 1:20
    numElements = size(element_Pos_Array_um_X, 2);
     % Lower and upper bounds only
    check = 1;

[II,delay] = PCIfreqRBCcuda_single(rfFrame, element_Pos_Array_um_X, ...
    speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, ...
    image_Range_X_um, image_Range_Z_um, 3, range_frq, check);
    
    arr_1_single{end+1} = delay;  % Use cell array to store if delay is not a scalar
end

disp('All delay values after 20 iterations for pthcoherence single: ');
disp(arr_1_single);
arr_1_double= [];

for i = 1:20
    numElements = size(element_Pos_Array_um_X, 2);
    check = 1;

[II,delay] = PCIfreqRBCcuda_double(rfFrame, element_Pos_Array_um_X, ...
    speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, ...
    image_Range_X_um, image_Range_Z_um, 3, range_frq, check);
    
    
    arr_1_double{end+1} = delay;  % Use cell array to store if delay is not a scalar
end

disp('All delay values after 20 iterations for pthcoherence Double: ');
disp(arr_1_double);

%% Interpolate
z = image_Range_Z_um * 1e-3;
x = image_Range_X_um * 1e-3;
[Xi, Zi] = meshgrid(x, z);

zz = linspace(z(1), z(end), numel(z)*2-1);
xx = linspace(x(1), x(end), numel(x)*2-1);
[Xf, Zf] = meshgrid(xx, zz);
Vqp4 = interp2(Xi, Zi, II', Xf, Zf, 'cubic');
Vqp4(Vqp4 < 0) = eps;

%% Plot
figure;
imagesc(xx, zz, 10*log10(Vqp4 ./ max(Vqp4(:))));
colorbar;
colormap('hot');
axis tight;
xlabel('Lateral Location (mm)');
ylabel('Range Location (mm)');
title('PCI Image (Single Frame)');
%%
% Convert to dB scale
log_image = 10 * log10(Vqp4 ./ max(Vqp4(:)));

% Clip to -15 dB dynamic range
log_image(log_image < -15) = -15;

% Plot
figure;
imagesc(xx, zz, log_image);
colorbar;
colormap('hot');
caxis([-15 0]);  % Fix color axis range explicitly
axis tight;

xlabel('Lateral Location (mm)');
ylabel('Range Location (mm)');
title('PCI Image (Single Frame, -15 dB Dynamic Range)');
