function [beamformed_Image] = PDAS(RData, element_Pos, speed_Of_Sound_umps, RF_Start_Time, fs, BeamformX, BeamformZ,element_loc,p)
beamformed_Image = zeros(length(BeamformX), length(BeamformZ));
%beamformedwCF_Image=zeros(length(image_Range_X_um), length(image_Range_Z_um));
S = size(RData);
N = S(1);
    for Xi = 1:length(BeamformX)
        for Zi = 1:length(BeamformZ)
            pixelAmp = 0;
            Coherence1=0;
            Coherence2=0;
                for ex = 1:length(element_Pos)
                    distance_Along_RF = sqrt(((BeamformX(Xi)-element_loc(1))^2)+BeamformZ(Zi)^2)+sqrt((BeamformX(Xi)- element_Pos(ex))^2 +(BeamformZ(Zi))^2);
                    time_Pt_Along_RF = distance_Along_RF/(speed_Of_Sound_umps);
                    sample_Pt = round((time_Pt_Along_RF-RF_Start_Time)*fs) +1;
                    if(sample_Pt>N || sample_Pt<1)
                      continue;
                    else
                        RFAmp = sign(RData(sample_Pt, ex))*(abs(RData(sample_Pt, ex))^(1/p));
                        CF1=(((RData(sample_Pt, ex))));
                        CF2=abs((RData(sample_Pt, ex)))^2;
                    end
                    pixelAmp = pixelAmp + RFAmp;
                    Coherence1=Coherence1+CF1;
                    Coherence2=Coherence2+CF2;
                end
            beamformed_Image(Xi,Zi) = sign(pixelAmp)*(abs(pixelAmp)^p);
        end
    end
end
