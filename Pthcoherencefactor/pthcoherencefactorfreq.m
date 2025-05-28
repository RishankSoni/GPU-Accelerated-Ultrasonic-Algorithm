function [pCF]=pthcoherencefactorfreq(RData,p)
[~,col]=size(RData);
MagRF_Arr=abs(RData);
CompressedImage=MagRF_Arr.^(1/p);
Phase_RFArr=angle(RData);
Signal=CompressedImage.*(exp(1i.*(Phase_RFArr)));
beamformed=abs(sum(Signal,2)).^p;
Nr=beamformed.^2;
dR=(abs(RData)).^2;
Dr=sum(dR,2);
Dr(Dr==0)=eps;
CF=(1/col).*(Nr./Dr);
pCF=((CF));
end