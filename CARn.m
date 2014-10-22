function [signalout] = CARn(signal, chanDim)
num_chans = chanDim;

signal = signal - repmat(mean(signal, 1), size(signal,1),1);

spatfiltmatrix=[];
spatfiltmatrix=eye(num_chans) - ones(num_chans)/num_chans;
signalout = double(signal) * spatfiltmatrix;