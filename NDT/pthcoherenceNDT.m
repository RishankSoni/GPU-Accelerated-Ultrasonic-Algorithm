function [BeamformData] = pthcoherenceNDT(RData, element_Pos, speed_Of_Sound_umps, RF_Start_Time, fs, BeamformX, BeamformZ,element_loc,p)
[~,~]=size(RData);
BeamformData=zeros(size(BeamformX,2),size(BeamformZ,2));
    % for Xi = 1:length(BeamformX)
    %     for Zi = 1:length(BeamformZ)
    %         sum=0;
    %             for ex = 1:length(element_Pos)
    %                 distance_Along_RF = sqrt(((BeamformX(Xi)-element_loc(1))^2)+BeamformZ(Zi)^2)+sqrt((BeamformX(Xi)- element_Pos(ex))^2 +(BeamformZ(Zi))^2); 
    %                 time_Pt_Along_RF = distance_Along_RF/(speed_Of_Sound_umps);
    %                 samples = round((time_Pt_Along_RF-RF_Start_Time)*fs) +1; 
    %         if samples>size(RData,1)|| samples<1
    %            continue;
    %         else
    %             temp(ex)=sign(RData(samples,ex)).*(abs(RData(samples,ex)));  
    %             sum=sum+temp(ex);
    %         end
    %             end  
    %     [pCF] = pthcoherencefactor(temp,p);
    %     BeamformData(Xi ,Zi)=sum*pCF; 
    %     end 
    % end
    % BeamformData = trail(RData, element_Pos, speed_Of_Sound_umps, RF_Start_Time, fs, BeamformX, BeamformZ, element_loc, p);
    BeamformData = trails( ...
    single(RData), single(element_Pos), single(speed_Of_Sound_umps), ...
    single(RF_Start_Time), single(fs), single(BeamformX), ...
    single(BeamformZ), single(element_loc), single(p));
end
