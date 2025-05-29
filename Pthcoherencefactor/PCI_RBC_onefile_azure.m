    % %% Clear workspace and add folder with data
    % close all;
    % clearvars;
    % 
    % % Prompt user to select the folder containing the data files
    % pn = uigetdir('Select folder containing the data files');
    % if pn == 0
    %     error('No folder selected.');
    % end
    % addpath(pn);
    % pp = dir(fullfile(pn, '*.mat'));
    % 
    % % Load the setup file (make sure it’s in the folder or current path)
    % setupFile = 'SetUpC5_2v_ChirpPCI_AP.mat';
    % if exist(setupFile, 'file')
    %     load(setupFile);
    % else
    %     error('Setup file not found.');
    % end
    % 
    % 
    % disp("Hello")
    % %% Select a dataset
    % % Look for a file whose name starts with 'PD'
    % datasetFile = '';
    % for j = 1:length(pp)
    %     if strncmp(pp(j).name, 'PD', 2)
    %         datasetFile = pp(j).name;
    %         break;
    %     end
    % end
    % 
    % % If no PD dataset is found, prompt the user to select a dataset file manually.
    % if isempty(datasetFile)
    %     [fileName, folderName] = uigetfile('*.mat', 'No PD dataset found. Please select a dataset file:');
    %     if isequal(fileName,0)
    %         error('No dataset selected.');
    %     end
    %     datasetFile = fullfile(folderName, fileName);
    % else
    %     datasetFile = fullfile(pn, datasetFile);
    % end

    %% Clear workspace and add folder with data
close all;
clearvars;

% Hardcoded path to the folder and dataset
pn = 'C:\Users\arjun\Desktop\Sem 4\ProjectCourse\Codes\RCB beamforming';
addpath(pn);
pp = dir(fullfile(pn, '*.mat'));

% Load the setup file
setupFile = fullfile(pn, 'SetUpC5_2v_ChirpPCI_AP.mat');
if exist(setupFile, 'file')
    load(setupFile);
else
    error('Setup file not found.');
end

disp("Hello")

%% Select a dataset (hardcoded)
datasetFile = fullfile(pn, 'PData_RBC2_2023Feb14_1_area_2_dataset_99_66338.mat');

% Check if it exists
if ~isfile(datasetFile)
    error('Specified dataset file not found: %s', datasetFile);
end


% Load the selected dataset (expects variable RData in the file)
load(datasetFile);  

%% Use only the first frame for processing
% RData is assumed to be a 3D array: samples x channels x frames.
numSamples = min(size(RData,1), 1472);  % Use up to 1472 samples, or fewer if available.
rfFrame = double(RData(1:numSamples, 1:128, 1));

%% Define imaging parameters
center_frequency = 3.68e6;         % Center frequency used for acquisition
wavelength = (1540e3 / center_frequency);
Aperture = Trans.ElementPos(:, [1 3])';  % Assumes variable 'Trans' is loaded from the setup file
Aperture = Aperture * wavelength;
element_Pos_Array_um_X = Aperture * 1e3; % Convert to micrometers

sampling_Freq = 4 * center_frequency;  % Sampling frequency
image_Range_X_um = linspace(-30.0901, 30.1707, 66) * 1e3;
image_Range_Z_um = linspace(19.91, 77.6644, 125) * 1e3;
speed_Of_Sound_umps = 1540e6;  % Speed of sound in um/s
RF_Start_Time = 0;           % Start time for RF data processing

%% Run PCI imaging on the selected frame
% Assumes the function PCIimagingSparseupdated is available in your MATLAB path.
range_frq = [2e6, 5e6];  % Lower and upper bounds only
check = 1;

II = PCIfreqRBCcuda(rfFrame, element_Pos_Array_um_X, ...
    speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, ...
    image_Range_X_um, image_Range_Z_um, 2, range_frq, check);

%% Interpolate for smoother display
z = image_Range_Z_um * 1e-3; % Convert to mm
x = image_Range_X_um * 1e-3; % Convert to mm
[Xi, Zi] = meshgrid(x, z);
zz = linspace(z(1), z(end), numel(z)*2-1);
xx = linspace(x(1), x(end), numel(x)*2-1);
[Xf, Zf] = meshgrid(xx, zz);
Vqp4 = interp2(Xi, Zi, II', Xf, Zf, 'cubic');
Vqp4(Vqp4 < 0) = eps;


%% Plot the resulting PCI image
figure;
imagesc(xx, zz, 10*log10(Vqp4./max(Vqp4(:))));
colorbar;
colormap('hot');

axis tight;
xlabel('Lateral Location (mm)');
ylabel('Range Location (mm)');
title('PCI Image (Single Frame)');

%%
Vqp4 = interp2(Xi, Zi, II', Xf, Zf, 'cubic');
Vqp4(isnan(Vqp4)) = eps;
Vqp4(Vqp4 < 0) = eps;
    
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
