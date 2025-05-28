function [beamformed_Image] = PCIfreqRBC(RF_Arr, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, image_Range_X_um, image_Range_Z_um,p,range_frq,check)
beamformed_Image = zeros(length(image_Range_X_um), length(image_Range_Z_um));
fk=linspace(0,sampling_Freq,size(RF_Arr,1));
[~,FlowerIndex]=min(abs(fk-range_frq(1)));
[~,FupperIndex]=min(abs(fk-range_frq(2)));
disp('Beam forming has been started for phantom,enjoy!');
% Var_wc=zeros(size(RF_Arr,1),size(RF_Arr,2));
% Var_wd=zeros(size(RF_Arr,1),size(RF_Arr,2));
    for Xi = 1:length(image_Range_X_um)
        Xi
        for Zi = 1:length(image_Range_Z_um)
            distance_Along_RF = sqrt(((image_Range_X_um(Xi)- element_Pos_Array_um_X(1,:)).^2) +((image_Range_Z_um(Zi)-element_Pos_Array_um_X(2,:)).^2)); 
            time_Pt_Along_RF = (distance_Along_RF/(speed_Of_Sound_umps));
            %temp_delay=delayseq(RF_Arr,repelem(RF_Start_Time,Col),sampling_Freq);
            %temp=delayseq(RF_Arr,-(time_Pt_Along_RF),sampling_Freq);
            [~,freqtemp]=simpledelayfreq(RF_Arr,-(time_Pt_Along_RF.*sampling_Freq)',sampling_Freq);
%%
           %% NLmagnitude scaled beamforming
            var=freqtemp(FlowerIndex:FupperIndex,:);  
            if check==1
               [pDAS]=pthrootfreq(var,1); %% change this 1 for pCF and p for pCF
               [CF]=pthcoherencefactorfreq(var,p); %% chnage this for pCF
               DCoffset=sum(abs(var.*CF).^2,2); %% DCoffset for pCF
               beamformed_Image(Xi,Zi)=sum(((pDAS.*(CF)).^2)-DCoffset); %% pCF
            else
               [pDAS]=pthrootfreq(var,p); %% 
               DCoffset=sum(abs(var).^2,2); %% DC offset for pDAS
               beamformed_Image(Xi,Zi)=sum(((pDAS).^2)-DCoffset); %% Pdas
            end

        end
    end