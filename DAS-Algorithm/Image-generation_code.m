%% DAS beamforming
clc
clear
close all

%%
% load('C:\Users\ruchi\Desktop\SR\Datacollectedwithhari\Simulation mode data\inbuiltdata2023-08-01-041511.mat');
 load('numberofanglesis32025-02-03-191625.mat');
unbeamformedRfdata = double(cell2mat(RcvData));
%% Output from Verasonics and visualization
for i=1:length(IDatasaved)
    I=IDatasaved{:,i};
    Q=QDatasaved{:,i};
    IQ(:,:,i)=complex(I,Q);
end
P.Wavelength = Resource.Parameters.speedOfSound/(Trans.frequency*1000);% for chirp pulse
PARAM.z = (PData(1).Origin(3)+[0:PData.Size(1)-1]*PData(1).PDelta(3))*P.Wavelength/1000;
PARAM.x = (PData(1).Origin(1)+[0:PData.Size(2)-1]*PData(1).PDelta(1))*P.Wavelength/1000;
[mesh_X,mesh_Z] = meshgrid(PARAM.x, PARAM.z);
% Convert IQ to B-mode
Bmode = 20*log10(abs(IQ));Bmode = squeeze(Bmode)-max(Bmode(:)); % convert complex IQ into Bmode images
dbmax = -30; % max [dB]
      im = imagesc(PARAM.x*1e3,PARAM.z*1e3,Bmode(:,:,1),[dbmax,0]);axis image; colormap gray;clb=colorbar;hold on
    tt = title(['B mode Normal Pulse Frame ' num2str(1)]);
xlabel('Lateral width (mm)'), ylabel('Axial width (mm)'); clb.Title.String='dB';
%% Beamforming parameters
addpath('C:\Users\arjun\Downloads\OneDrive_1_1-22-2025\MUST');
% addpath('C:\Dataset\ruchika_codes\Data and codes given by the authors\MUST');
PARAM.bandwidth = (Trans.Bandwidth(2)-Trans.Bandwidth(1))/Trans.frequency * 100; %bandwidth [% f0]
PARAM.fc = Trans.frequency*1e6; %central frequency [Hz]
PARAM.fs = 31.25*1e6; % sampling frequency (100% bandwidth mode of Verasonics) [Hz]Receive.demodFrequency
PARAM.c = Resource.Parameters.speedOfSound; % speed of sound [m/s]
PARAM.wavelength = PARAM.c/PARAM.fc; % Wavelength [m]
PARAM.xe = Trans.ElementPos(:,1)*PARAM.wavelength; % x coordinates of transducer elements [m]
PARAM.Nelements = Trans.numelements; %number of transducers
%   PARAM.t0 = 2*P.startDepth*PARAM.wavelength/PARAM.c - TW.peak/PARAM.fc; %time between the emission and the beginning of reception [s] % for chirp
   PARAM.t0 = 0; % for normal pulse and fundamentally matched chirp
P.numTx = 3; % number of transmit angles used in flash angles code
 P.Wavelength = Resource.Parameters.speedOfSound/(Trans.frequency*1000);% for chirp pulse
P.NDsample = Receive(1).endSample;%%--1920 number of samples obtained
angles_list = cat(1,TX.Steer);angles_list = angles_list(1:P.numTx,1);
PARAM.angles_list = angles_list; % list of angles [rad] (in TX.Steer)
PARAM.fnumber = 1.7;% MUST toolbox % fnumber;1.9;1.7
PARAM.compound = 1; % flag to compound [1/0]

% Pixels grid (extracted from PData), in [m] 
PARAM.z = (PData(1).Origin(3)+[0:PData.Size(1)-1]*PData(1).PDelta(3))*P.Wavelength/1000;
PARAM.x = (PData(1).Origin(1)+[0:PData.Size(2)-1]*PData(1).PDelta(1))*P.Wavelength/1000;
[mesh_X,mesh_Z] = meshgrid(PARAM.x, PARAM.z);
clear angles_list
%% Beamforming
IQ = zeros([PData.Size(1:2) 1+(~PARAM.compound)*(P.numTx-1)],'single');
tic;
numberofframes = size(unbeamformedRfdata,3);
for ii = 1:numberofframes% number of frames
       for iTX = 1:P.numTx 
           this_RF = single(RcvData{1}(Receive(iTX).startSample:Receive(iTX).endSample,:,ii)); 
                RF_IQ{iTX} = this_RF;
       end
       % Delay And Sum beamforming
    IQ(:,:,:,ii) = BF_DelayAndSum(RF_IQ, PARAM, mesh_X, mesh_Z);
    clc;
    disp(ii)
end
t_end=toc;clear RFk SIG IQ_i ii %RFdata
disp([num2str(size(IQ,4)) ' frames beamformed in ' num2str(round(t_end,1)) ' sec.'])

%% Find dynamic range of beamformed RF
% Convert pixel values to double
IQ_new = double(abs(IQ));
% Calculate the dynamic range
min_value = min(IQ_new(:));
max_value = max(IQ_new(:));
 % Add a small offset to avoid division by zero
offset = 1e-6; % Small offset value
min_value = min_value + offset;
% Calculate the dynamic range in dB
dynamic_range_db = 20 * log10(max_value / min_value);
dynamic_range_db = abs(dynamic_range_db);
%% Average all the frames
 IQ = reshape(IQ, [],128,100);
%%
r = size(IQ,3);
IQ_2 = abs(IQ);
IQ_avg = sum(IQ,3)./r;
%% Find dynamic range of data
% Convert pixel values to double
IQ_new = double(IQ_avg(:,:,1));
% Calculate the dynamic range
min_value = min(IQ_new(:));
max_value = max(IQ_new(:));
% Calculate the dynamic range in dB
dynamic_range_db = 20 * log10(max_value / min_value);
dynamic_range_db = abs(dynamic_range_db);
%% Displaying IQ converted in Bmode scale
figure();clf
 Bmode = 20*log10(abs(IQ));Bmode = squeeze(Bmode)-max(Bmode(:)); % convert complex IQ into Bmode images
dbmax = -30; % max [dB]
      im = imagesc(PARAM.x*1e3,PARAM.z*1e3,Bmode(:,:,1),[dbmax,0]);axis image; colormap gray;clb=colorbar;hold on
    tt = title(['B mode Normal Pulse Frame ' num2str(1)]);
xlabel('Lateral width (mm)'), ylabel('Axial width (mm)'); clb.Title.String='dB';

%%
function IQ = BF_DelayAndSum(SIG, PARAM, X, Z)
%% BF_DelayAndSum: Delay-And-Sum beamforming with compounding.
%  SIG: a cell array of RF (or RF_IQ) signals (each element: [nSamples x Nelements])
%  PARAM: structure containing necessary parameters (f0, fs, t0, c, xe, etc.)
%  X, Z: 2D imaging grids from meshgrid.
%
%  The mex file BF_das_rephaseSignal (compiled from CUDA code) is expected to
%  return a vector of length numel(X) when given a 2D SIG. We then reshape it to
%  match the grid.
%
%  This version works for both compound (summing over angles) or noncompound
%  (storing each angle separately).

IQ = zeros(size(X), 'double');  % initialize output as a 2D array

if ~isfield(PARAM, 'compound')
    PARAM.compound = 1;
end
PARAM.p_pDAS = 2;
% (Make sure PARAM.f0 exists; here we force a value if missing.)
if ~isfield(PARAM, 'f0')
    PARAM.f0 = 5e6;  % example central frequency (Hz)
end
if ~isfield(PARAM, 'theta')
    PARAM.theta = PARAM.angles_list(1);
end

% Optionally, test one call and display its size.
% testOut = BF_das_rephaseSignal(double(abs(SIG{1})), PARAM, X, Z);
% disp('Size of output from mex function (before reshape):');
% disp(size(testOut));
% disp('Size of grid X:');
% disp(size(X));
% For each angle, call the mex function and add (or store) the result.
disp("hello")
disp(((SIG(1))));

for k = 1:length(PARAM.angles_list)
    
    PARAM.theta = PARAM.angles_list(k);
    % Call the mex function.
    temp = BF_das_rephaseSignal(double(SIG{k}), PARAM, X, Z);
    % In case the mex file returns a column vector (numel(X) x 1), reshape it.
    if ~isequal(size(temp), size(X))
        temp = reshape(temp, size(X));
    end

    if PARAM.compound
        IQ = IQ + temp;
    else
        % If not compounding, store each beamformed image in a third dimension.
        IQ(:, :, k) = temp;
    end
end

end



%%
