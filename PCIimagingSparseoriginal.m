function [beamformed_Image] = PCIimagingSparseupdated(RF_Arr, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, image_Range_X_um, image_Range_Z_um,p)
beamformed_Image = zeros(length(image_Range_X_um), length(image_Range_Z_um));
%%% written by Abhinav Singh
[Axial_depth,~] = size(RF_Arr);
N = Axial_depth;
disp('Beam forming has been started for phantom,enjoy!');
    for Xi = 1:length(image_Range_X_um)
        Xi
        for Zi = 1:length(image_Range_Z_um)
            distance_Along_RF = sqrt(((image_Range_X_um(Xi)- element_Pos_Array_um_X(1,:)).^2) +((image_Range_Z_um(Zi)-element_Pos_Array_um_X(2,:)).^2)); 
            time_Pt_Along_RF = (distance_Along_RF/(speed_Of_Sound_umps));
            % [temp]=onlyfft(RF_Arr,-(time_Pt_Along_RF.*sampling_Freq)');
            % [pDAS]=p_cuda(temp,p);
            delay = -(time_Pt_Along_RF.*sampling_Freq)';
            [fftBin, RFFT, temp, pDAS] = combined_cuda(RF_Arr, delay, p);

            DCoffset=sum(abs(temp).^2,2); %% change this for DC offset
            beamformed_Image(Xi,Zi)=sum(((pDAS)'.^2)-DCoffset); %%% change
       
        end
    end