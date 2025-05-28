function [beamformed_Image] = PCIinvivofreq(RF_Arr, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, image_Range_X_um, image_Range_Z_um,p,range_frq,check)
beamformed_Image = zeros(length(image_Range_X_um), length(image_Range_Z_um));
%beamformedwCF_Image=zeros(length(image_Range_X_um), length(image_Range_Z_um));
disp('Beam forming has been started for pig..Updated>>>>>>>>');
fk=linspace(0,sampling_Freq,size(RF_Arr,1));
[~,FlowerIndex]=min(abs(fk-range_frq(1)));
[~,FupperIndex]=min(abs(fk-range_frq(2)));
 for Xi = 1:length(image_Range_X_um)
        Xi
        for Zi = 1:length(image_Range_Z_um)        
            distance = sqrt((image_Range_X_um(Xi)- element_Pos_Array_um_X).^2 +(image_Range_Z_um(Zi))^2); 
            time = (distance/(speed_Of_Sound_umps));
            [~,freqtemp]=simpledelayfreq(RF_Arr,-(time.*sampling_Freq)',sampling_Freq);
%% NLmagnitude scaled beamforming
           var=freqtemp(FlowerIndex:FupperIndex,:);                   
           if check==1 
            [CF]=pthcoherencefactorfreq(var,p);
            [pDAS]=pthrootfreq(var,1);
            DCoffset=sum(abs(var.*CF).^2,2);
            beamformed_Image(Xi,Zi)=sum(((pDAS.*(CF)).^2)-DCoffset);
            %
           else
               [pDAS]=pthrootfreq(var,p);
               DCoffset=sum(abs(var).^2,2); %% Pth root
               beamformed_Image(Xi, Zi)=sum((((pDAS).^2)-DCoffset)); %% pth
           end
            %root
            %
            %[psi]=RobustCaponBeamforming(R,p,n);
            %beamformed_Image(Xi,Zi)=psi;
       end
end