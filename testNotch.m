% Fs = 1000;
% N = 1000;
% i = [0 : N-1]';
Fs = 600;
OFFSET = 600*20;
nfft=4096;
time= (1 + OFFSET:OFFSET + (600 * 20));

Hd = equiripple_filter_brainnet;
SELECTED = CARn(data, 12);
SELECTED = filtfilt(Hd.Numerator,1, SELECTED);
dPOz = SELECTED(time, 4);

x0 = cos(2*pi*5.6*time / Fs);  
x1 = sin(2*pi*5.6*time / Fs);
x2 = cos(2*pi*6.4*time / Fs);  
x3 = sin(2*pi*6.4*time / Fs);

%compute corrcoef
prima = [];
[a b r] = canoncorr(dPOz, x0' + x1');
prima = [r prima];
[a b r] = canoncorr(dPOz, x2' + x3');
prima = [r prima];

%create the reference signal of the adaptive filter


%notch filter architecture
wo = 6.4/(Fs/2);  bw = wo/35;
[b,a] = iirnotch(wo,bw);
e = filtfilt(b, a, dPOz);

dopo = [];
[a b r] = canoncorr(e, x0' + x1');
dopo = [r dopo];
[a b r] = canoncorr(e, x2' + x3');
dopo = [r dopo];

%compute the spectrum of the initial signal and the filtered signal
[Pw, Fw]=periodogram(dPOz,hann(size(time,2)),nfft,Fs);
[Pw2, Fw2]=periodogram(e,hann(size(time,2)),nfft,Fs);
figure(1)
s1 = subplot(2,1,1);
plot(Fw, Pw);
title(s1, 'Bipolar');
s2 = subplot(2,1,2);
plot(Fw2, Pw2);
title(s2, 'Filtered');