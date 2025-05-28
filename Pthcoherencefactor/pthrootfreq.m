function [pDAS]=pthrootfreq(var,p)
MagRF_Arr=abs(var);
CompressedImage=MagRF_Arr.^(1/p);
Phase_RFArr=angle(var);
Signal=CompressedImage.*(exp(1i.*(Phase_RFArr)));
pDAS=abs(sum(Signal,2)).^p;
%pDAS=pDAS';
end