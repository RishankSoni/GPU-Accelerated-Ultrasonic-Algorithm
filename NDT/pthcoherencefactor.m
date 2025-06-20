function [pCF] = pthcoherencefactor(Rdata,p)
%%% pth-Coherencefactor!
[~,col]=size(Rdata);
nR=sign(sum(sign(Rdata).*(abs(Rdata).^(1/p)),2)).*((abs(sum(sign(Rdata).*(abs(Rdata).^(1/p)),2))).^p);
%nR=((sum(Rdata,2)).^2);
Nr=(nR).^2;
%Nr=nR;
dR=(abs(Rdata)).^2;
Dr=sum(dR,2);
Dr(Dr==0)=eps;
CF=(1/col).*(Nr./Dr);
pCF=((CF))';
end