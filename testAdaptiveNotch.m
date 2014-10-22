% Fs = 600;
% desp = 4;
% OFFSET = 600 * 20;
% nfft=4096;
% time= (1 + OFFSET:OFFSET + (600 * 20));
% 
% Hd = equiripple_filter_brainnet;
% SELECTED = CARn(data, 12);
% SELECTED = filtfilt(Hd.Numerator,1, SELECTED);

% creo referenze
% PROVA METTERE COSENO
ref1 = sin(2 * pi * 6.4 * time / Fs);
ref2 = sin(2 * pi * 6.9 * time / Fs);

% imposto i due filtri notch

% filtro i dati

% comparo i coefficienti

Fs = 1000;
N = 1000;
i = [0 : N-1]';

%create the initial signal
x = sin(2*pi*200* i/Fs) + 0.66*sin(2*pi* 280*i/Fs) + 0.59*sin(2*pi*60*i/Fs) + 0.5^0.5*randn( N,1);

%create the reference signal of the adaptive filter
x0 = cos(2*pi*6.4*i / Fs);  
x1 = sin(2*pi*6.4*i / Fs);

%adaptive filter architecture
L = 2;
step_size = 0.005;
w = zeros(1,L);

%run the adaptive filter


for i = L:length(time)
    temp_x1 = x1(i-1:i-L+1); 
    temp_x2 = x2(i-1:i-L+1);


    yk1 = sum(w(1).*temp_x1);
    yk2 = sum(w(2).*temp_x2);

    
    y(i) = yk1+yk2;

     e = x(i)-y(i); 

    for j=(1:L)
             w(j)=w(j)+2*step_size*e*temp_x1(j);
             w(j)=w(j)+2*step_size*e*temp_x2(j);

            end
    end

%compute the spectrum of the initial signal and the filtered signal
f = [0 : Fs/N : Fs - Fs/N]';
F = abs(fft(x));
E = abs(fft(e));

%plot
figure;
subplot(411) ;plot(x); title('initial signal');
subplot(412) ;plot(e); title('initial signal after filtering');
subplot(413) ;plot(f,F( 1:length( f)));title( 'spectrum of initial signal');
subplot(414) ;plot(f,E( 1:length( f)));title( 'spectrum of initial signal after filtering'); 