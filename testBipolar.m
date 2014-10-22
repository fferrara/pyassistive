% Configurazione prova
% Miglior canale bipolare vs filtro adattativo con stesso canale non
% stimolato
% Filtro adattativo vince con finestra di 2s

Fs = 600;
desp = 3;
nfft=4096;

Hd = equiripple_filter_brainnet;
SELECTED = CARn(data, 12);
SELECTED = filtfilt(Hd.Numerator,1, SELECTED);

NOISE = CARn(data2, 12);
NOISE = filtfilt(Hd.Numerator, 1, NOISE);


% ref


dNOISE = NOISE(1:Fs*desp, 4);

for i = 0:17
    OFFSET = 600 * i;
    WINDOW = (1 + OFFSET:OFFSET + desp * Fs);
    
X = [];
x0 = cos(2*pi*5.6*WINDOW / Fs) + sin(2*pi*5.6*WINDOW / Fs);  
x1 = cos(2*pi*2*5.6*WINDOW / Fs) + sin(2*pi*2*5.6*WINDOW / Fs); 
x2 = cos(2*pi*6.4*WINDOW / Fs) + sin(2*pi*6.4*WINDOW / Fs);
x3 = cos(2*pi*2*6.4*WINDOW / Fs) + sin(2*pi*2*6.4*WINDOW / Fs);
X = [X;x0;x1;x2;x3];
    
    % Bipolar: O1 - POz, O2 - POz, Oz - POz
%     dO2 = SELECTED(time, 11);
%     dOz = SELECTED(time, 12);
    dPOz = SELECTED(WINDOW, 4);
    dO1 = SELECTED(WINDOW, 10);

    mu = 0.01;          % Set the step size for algorithm updating.
    ha = adaptfilt.nlms(20,mu);
    [y,e] = filter(ha,dNOISE,dPOz);

    [Pw, Fw]=periodogram(dPOz - dO1,hann(size(WINDOW,2)),nfft,Fs);
    [Pw2, Fw2]=periodogram(e,hann(size(WINDOW,2)),nfft,Fs);
    [Pw3, Fw3] = periodogram(dPOz, hann(size(WINDOW,2)),nfft,Fs);
    figure(1)
    s1 = subplot(2,2,1);
    plot(Fw, Pw);
    title(s1, 'Bipolar');
    s2 = subplot(2,2,2);
    plot(Fw2, Pw2);
    title(s2, 'Filtered');
    s3 = subplot(2,2,3);
    plot(Fw3, Pw3);
    title(s3, 'Monopolar');

    ros1 = [];
    ros2 = [];
    for j=1:4
        [~, ~, r1] = canoncorr(dPOz - dO1, X(j,:)');
        [~, ~, r2] = canoncorr(e, X(j, :)');
        ros1 = [r1 ros1];
        ros2 = [r2 ros2];
    end
    ros1 = reshape(ros1,2,[]);
    ros2 = reshape(ros2,2,[]);
    
    [foo bar] = max(max(ros1(:,:)));
    out1 = bar
    [foo bar] = max(max(ros2(:,:)));
    out2 = bar
end
