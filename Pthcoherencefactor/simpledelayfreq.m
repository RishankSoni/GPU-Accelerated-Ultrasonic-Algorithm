function [ShiftData_time,ShiftData]=simpledelayfreq(RData,delay,fs)
nfft=(size(RData,1));
binStart = floor(nfft/2);
fftBin = (2*pi*ifftshift(((0:nfft-1)-binStart).'))/nfft; 
fftBin=fftBin';
%fk = linspace(-fs/2,fs/2,size(RData,1))./size(RData,1);
RFFT=fft(RData,nfft,1);
ShiftData = RFFT.*((exp(-1i*delay*fftBin)).'); 
ShiftData_time=ifft(ShiftData,nfft,1);
ShiftData_time=real(ShiftData_time);
end 