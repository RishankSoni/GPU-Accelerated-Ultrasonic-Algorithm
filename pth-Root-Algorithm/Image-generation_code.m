%% DAS beamforming
clc
clear
 close all
 %% Load the data
 load('inbuiltdata2023-08-01-041511.mat')
unbeamformedRfdata = double(cell2mat(RcvData));
%% Beamforming parameters
PARAM.bandwidth = (Trans.Bandwidth(2)-Trans.Bandwidth(1))/Trans.frequency * 100; %bandwidth [% f0]
PARAM.fc = Trans.frequency*1e6; %central frequency [Hz]
PARAM.fs = 31.25*1e6; % sampling frequency (100% bandwidth mode of Verasonics) [Hz]Receive.demodFrequency
PARAM.c = Resource.Parameters.speedOfSound; % speed of sound [m/s]
PARAM.wavelength = PARAM.c/PARAM.fc; % Wavelength [m]
PARAM.xe = Trans.ElementPos(:,1)*PARAM.wavelength; % x coordinates of transducer elements [m]
PARAM.Nelements = Trans.numelements; %number of transducers
PARAM.t0 = 0;
P.numTx = 1; % number of transmit angles used in flash angles code
P.Wavelength = Resource.Parameters.speedOfSound/(TW.Parameters(1)*1000);
P.NDsample = Receive(1).endSample;%%--1920
%--//End of Revision
angles_list = cat(1,TX.Steer);angles_list = angles_list(1:P.numTx,1);
PARAM.angles_list = angles_list; % list of angles [rad] (in TX.Steer)
PARAM.fnumber = 1.71;% MUST toolbox % fnumber;1.9
PARAM.compound = 0; % flag to compound [1/0]
PARAM.p_pDAS = 2;
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
       for iTX = 1:P.numTx % convert 100 bandiwdth data to IQ signal
           this_RF = double(RcvData{1}(Receive(iTX).startSample:Receive(iTX).endSample,:,ii)); 
            PARAM.SIG_list{iTX} = this_RF;
       end
       
       % Delay And Sum beamforming
    IQ(:,:,:,ii) = BF_pDAS(PARAM.SIG_list, PARAM, mesh_X, mesh_Z);
    clc;
    disp(ii)
end
t_end=toc;clear RFk SIG IQ_i ii %RFdata
disp([num2str(size(IQ,4)) ' frames beamformed in ' num2str(round(t_end,1)) ' sec.'])
%% Displaying IQ converted in Bmode scale
figure();clf
Bmode = 20*log10(abs(IQ(:,:,1,:)));Bmode = squeeze(Bmode)-max(Bmode(:)); % convert complex IQ into Bmode images
dbmax = -60; % max [dB]
im = imagesc(PARAM.x*1e3,PARAM.z*1e3,Bmode(:,:,1),[dbmax 0]);axis image; colormap(gray);clb=colorbar;hold on
tt = title(['Frame ' num2str(1)]);
xlabel('x [mm]'), ylabel('z [mm]'); clb.Title.String='dB';
%%
function IQ = BF_pDAS(SIG, PARAM, X, Z)
%% function IQ = BF_MinimumVariance(SIG, PARAM, X, Z)
% pDAS beamforming with compounding or not
% INPUTS:
%   - SIG: RF or IQ signal matrix

%   - PARAM: structure containing:
%       - f0: central frequency of US wave
%       - fs: sampling frequency
%       - t0: start time of receiving signal
%       - c: speed of sound in the medium
%       - Nelements: number of elements
%       - pitch: distance between 2 centers of element
%       - width: width of 1 element
%       - xe: position of the elements center on the x axis in mm (0 is the
%             middle of the axis)
%       - fnumber: f-number given by the function F_fnumber
%       - theta: angle of emission in radian
%       - compound: set to 1 if you want to do compounding, 0 else
%       - angles_list: list of angles if you do compounding
%       - SIG_list: dictionnary with different signals obtained with the
%                   angles in angles_list
%       - p_pDAS: power p of pDAS

%   - X, Z: grid on pixel (use meshgrid)

% OUTPUTS:
%   - IQ: Beamformed image (in linear scale, you need to log compress it)

% Alexandre Corazza, 13/10/2021

BF = zeros(size(X));
% epsilon_0 = 1e-30; %avoid to divide by 0

if ~isfield(PARAM, 'compound')
    PARAM.compound = 0;
end

if ~isfield(PARAM, 'compound')
    PARAM.compound = 1;
end

% (Make sure PARAM.f0 exists; here we force a value if missing.)
if ~isfield(PARAM, 'f0')
    PARAM.f0 = 5e6;  % example central frequency (Hz)
end
if ~isfield(PARAM, 'theta')
    PARAM.theta = PARAM.angles_list(1);
end
if ~PARAM.compound
    for k = 1:length(PARAM.angles_list)
        PARAM.theta = PARAM.angles_list(k);
        temp = BF_das_rephaseSignalpth(double(SIG{k}), PARAM, X, Z);
     if ~isequal(size(temp), size(X))
        temp = reshape(temp, size(X));
    end
        thisBF=temp;
        % thisBF = BF_das_rephaseSignal_pth(PARAM.SIG_list{k}, PARAM, X, Z);
           BF = sign(thisBF) .* (abs(thisBF).^PARAM.p_pDAS);
    end
        
elseif PARAM.compound %Compounding
    for k = 1:length(PARAM.angles_list)
        PARAM.theta = PARAM.angles_list(k);
        temp = BF_das_rephaseSignalpth(double(SIG{k}), PARAM, X, Z);
     if ~isequal(size(temp), size(X))
        temp = reshape(temp, size(X));
    end
        thisBF=temp;
       % thisBF = BF_das_rephaseSignal_pth(PARAM.SIG_list{k}, PARAM, X, Z);
       BF = BF + sign(thisBF) .* (abs(thisBF).^PARAM.p_pDAS);
    end
end

IQ = BF;   
end
% function migSIG1 = BF_das_rephaseSignal(SIG, PARAM, X, Z)
% % function migSIG1 = BF_das_rephaseSignal(SIG, PARAM, X, Z)
% %
% % Build the 3D matrix of rephased signals
% %
% % INPUTS:
% %   - SIG: RF or RF_IQ signal matrix
% %
% %   - PARAM: structure containing:
% %       - f0: central frequency of US wave
% %       - fs: sampling frequency
% %       - t0: start time of receiving signal
% %       - c: speed of sound in the medium
% %       - Nelements: number of elements
% %       - pitch: distance between 2 centers of element
% %       - width: width of 1 element
% %       - fnumber:
% %       - theta: angle of emission in radian
% %       - compound: set to 1 if you want to do compounding, 0 else
% %       - angles_list: list of angles if you do compounding
% %       - win_apod: apodization at the reception
% %
% %   - X, Z: grid on pixel (use meshgrid)
% %
% % OUTPUTS:
% %   - migSIG1: Beamformed image (in linear scale, you need to log compress it)
% %
% % Alexandre Corazza (13/10/2021)
% % inspired from the function "dasmtx" of MUST toolbox, Damien Garcia http://www.biomecardio.com
% 
% epsilon_0 = 1e-30; %avoid to divide by 0
% 
% if iscell(SIG)
%     SIG_class = class(SIG{1});
% else,SIG_class = class(SIG);
% end
% 
% migSIG = zeros([1 numel(X)],SIG_class);
% BF = zeros(size(X));
% % emit delay
%  TXdelay = (1/PARAM.c)*tan(PARAM.theta)*abs(PARAM.xe - PARAM.xe(1));
% 
% %source virtuelle
% % beta = 1e-8;
% % L = PARAM.xe(end)-PARAM.xe(1);
% % vsource = [-L*cos(PARAM.theta).*sin(PARAM.theta)/beta, -L*cos(PARAM.theta).^2/beta];
% 
% for k = 1:PARAM.Nelements
%     % dtx = sin(PARAM.theta)*X(:)+cos(PARAM.theta)*Z(:); %convention FieldII
%     % dtx = sin(PARAM.theta)*X(:)+cos(PARAM.theta)*Z(:) + mean(TXdelay)*PARAM.c; %convention FieldII
%     dtx = sin(PARAM.theta)*X(:)+cos(PARAM.theta)*Z(:) + mean(TXdelay-min(TXdelay))*PARAM.c; %convention Verasonics
%    % dtx = hypot(X(:)-vsource(1), Z(:)-vsource(2)) - hypot((abs(vsource(1))-L/2)*(abs(vsource(1))>L/2), vsource(2)); %source virtuelle, convention Verasonics
%     drx = hypot(X(:)-PARAM.xe(k), Z(:));
% 
%     tau = (dtx+drx)/PARAM.c;
% 
%     %-- Convert delays into samples
%     idxt = (tau-PARAM.t0)*PARAM.fs + 1;
%     I = idxt<1 | idxt>(size(SIG,1)-1);
%     idxt(I) = 1; % arbitrary index, will be soon rejected
% 
%     idx  = idxt;       % Not rounded number of samples to interpolate later
%     idxf = floor(idx); % rounded number of samples
%     IDX  = repmat(idx, [1 1 size(SIG,3)]); %3e dimension de SIG: angles
% 
%     %-- Recover delayed samples with a linear interpolation
%     TEMP = SIG(idxf,k,:).*(idxf+1-IDX) + SIG(idxf+1,k,:).*(IDX-idxf);
% 
%     TEMP(I,:) = 0;
% 
%     if (~isreal(TEMP)) % if IQ signals, rephase
%         TEMP = TEMP.*exp(2*1i*pi*PARAM.fc*tau);
%     end
%     weights = abs(TEMP+epsilon_0) .^ ((1-PARAM.p_pDAS)/PARAM.p_pDAS);
%     TEMP = sum(TEMP .* weights, 3);
%     % Fnumber mask
%     mask_Fnumber = abs(X-PARAM.xe(k)) < Z/PARAM.fnumber/2;    
%     migSIG = migSIG(:)+TEMP(:).*mask_Fnumber(:);
%     BF = migSIG;
% end
% migSIG1=reshape(BF, size(X));
% end
