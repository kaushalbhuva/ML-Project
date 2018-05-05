%% This code is derived from FeatExtract_1.m
%%% This code additionally is for writing or creating the training set
%%% Figure displays have been commented out
%%% File reading, feature extraction, file writing is done here

clearvars

%% Loop for reading a raw file, extracting features and writing to the train file

for i = 1:111
%% File name creator
    test = ['s' num2str(i) '.xlsx'];

%% Reading a data file

dat = importdata(test);

%% Columns 9,10,11 contain relevant data
%%% 9,10,11 ~ z,y,x

AccZ = dat.data(:,9);
AccY = dat.data(:,10);
AccX = dat.data(:,11);

% %%% Observe
% figure;subplot(3,1,1);plot(AccX,'r');subplot(3,1,2);plot(AccY,'g');subplot(3,1,3);plot(AccZ);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PreProcessing
%% Mean Zero

AccX = AccX - mean(AccX);
AccY = AccY - mean(AccY);
AccZ = AccZ - mean(AccZ);

% %%% Observe
% figure;subplot(3,1,1);plot(AccX,'r');subplot(3,1,2);plot(AccY,'g');subplot(3,1,3);plot(AccZ);

%% Signal energy

EnerX = AccX'*AccX; %----- FEATURE
EnerY = AccY'*AccY; %----- FEATURE
EnerZ = AccZ'*AccZ; %----- FEATURE

%% Signal Power

PowX = EnerX/length(AccX); %----- FEATURE
PowY = EnerY/length(AccY); %----- FEATURE
PowZ = EnerZ/length(AccZ); %----- FEATURE

%% Signal Instatneous power

PowX_I = AccX.^2;
PowY_I = AccY.^2;
PowZ_I = AccZ.^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Temporal information %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Filter the signals 

FIRfilt = ones(1,15)/15; 

FiltAccX = conv(AccX,FIRfilt);
FiltAccY = conv(AccY,FIRfilt);
FiltAccZ = conv(AccZ,FIRfilt);

% %%% Observe
% figure;subplot(3,1,1);plot(FiltAccX,'r');subplot(3,1,2);plot(FiltAccY,'g');subplot(3,1,3);plot(FiltAccZ);

%% Filter the instantaneous power signals

FIRfilt = ones(1,20)/20; % length to be decided 

FiltPowX_I = conv(PowX_I,FIRfilt,'same');
FiltPowY_I = conv(PowY_I,FIRfilt,'same');
FiltPowZ_I = conv(PowZ_I,FIRfilt,'same');

% %%% Observe
% figure;subplot(3,1,1);plot(FiltPowX_I,'r');subplot(3,1,2);plot(FiltPowY_I,'g');subplot(3,1,3);plot(FiltPowZ_I);

%%% Filtered instantatneous power signals show more promise (atleast visually)

%% Filtered the instantaneous power signals (Mean width)

% Halfsecsamp = 10; % To be decided statistically

[PKSX,LOCS,W] = findpeaks(FiltPowX_I);
W = sort(W,'descend');                         
MeanW_X = mean(W(1:10)); % To decide/justify 10  %----- FEATURE
[PKSY,LOCS,W] = findpeaks(FiltPowY_I);
W = sort(W,'descend');
MeanW_Y = mean(W(1:10));                         %----- FEATURE
[PKSZ,LOCS,W] = findpeaks(FiltPowZ_I);
W = sort(W,'descend');
MeanW_Z = mean(W(1:10));                         %----- FEATURE

%% Filtered the instantaneous power signals (Mean rise time (approx))

% Calculating slope (first difference)
diffX = diff(FiltPowX_I);
diffY = diff(FiltPowY_I);
diffZ = diff(FiltPowZ_I);

% First difference amplifies noise so filtering again
filtdiffX = conv(diffX,FIRfilt);
filtdiffY = conv(diffY,FIRfilt);
filtdiffZ = conv(diffZ,FIRfilt);
       
offset = 50;

% Get the positive slopes and negative slopes
% X
filtdiffXp = filtdiffX;
filtdiffXp(filtdiffX>offset)=1;filtdiffXp(filtdiffX<=offset)=0;
MeanRE_X = sum(filtdiffXp)/groupOnes(filtdiffXp); %----- FEATURE
% %%% Observe
% figure;stem(filtdiffXp)

filtdiffXm = filtdiffX;
filtdiffXm(filtdiffX<-offset)=1;filtdiffXm(filtdiffX>=-offset)=0;
MeanFE_X = sum(filtdiffXm)/groupOnes(filtdiffXm); %----- FEATURE
% %%% Observe
% figure;stem(filtdiffXm)


% Y
filtdiffYp = filtdiffY;
filtdiffYp(filtdiffY>offset)=1;filtdiffYp(filtdiffY<=offset)=0;
MeanRE_Y = sum(filtdiffYp)/groupOnes(filtdiffYp); %----- FEATURE
% %%% Observe
% figure;stem(filtdiffYp)

filtdiffYm = filtdiffY;
filtdiffYm(filtdiffY<-offset)=1;filtdiffYm(filtdiffY>=-offset)=0;
MeanFE_Y = sum(filtdiffYm)/groupOnes(filtdiffYm); %----- FEATURE
% %%% Observe
% figure;stem(filtdiffYm)

% Z
filtdiffZp = filtdiffZ;
filtdiffZp(filtdiffZ>offset)=1;filtdiffZp(filtdiffZ<=offset)=0;
MeanRE_Z = sum(filtdiffZp)/groupOnes(filtdiffZp); %----- FEATURE
% %%% Observe
% figure;stem(filtdiffZp)

filtdiffZm = filtdiffZ;
filtdiffZm(filtdiffZ<-offset)=1;filtdiffZm(filtdiffZ>=-offset)=0;
MeanFE_Z = sum(filtdiffZm)/groupOnes(filtdiffZm); %----- FEATURE
% %%% Observe
% figure;stem(filtdiffZm)



% %%% Observe
% figure;plot(filtdiffX,'r');hold on;
%        plot(filtdiffY,'g');hold on;
%        plot(filtdiffZ);hold off

%% Filtered the instantaneous power signals (Mean fall time (approx))



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Frequency domain information %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

fs = 20; % 20 Hz sampling
nfft = length(AccX);
Obstime = 10; % 10 seconds (freq res possible 0.1 Hz)

fftFiltPowX_I = fft(FiltPowX_I);
fftFiltPowY_I = fft(FiltPowY_I);
fftFiltPowZ_I = fft(FiltPowZ_I);

% Pick the first and second dominant frequencies as features
tempfft = abs(fftshift(fftFiltPowX_I));
[PKSfftX,LOCSfftX] = findpeaks(tempfft(102:end),'Npeaks',2);  %----- FEATURE %----- FEATURE
LOCSfftX = LOCSfftX * fs/nfft;                                   %----- FEATURE %----- FEATURE

tempfft = abs(fftshift(fftFiltPowY_I));
[PKSfftY,LOCSfftY] = findpeaks(tempfft(102:end),'Npeaks',2);  %----- FEATURE %----- FEATURE
LOCSfftY = LOCSfftY * fs/nfft;                                    %----- FEATURE %----- FEATURE

tempfft = abs(fftshift(fftFiltPowZ_I));
[PKSfftZ,LOCSfftZ] = findpeaks(tempfft(102:end),'Npeaks',2);  %----- FEATURE %----- FEATURE
LOCSfftZ = LOCSfftZ * fs/nfft;                                     %----- FEATURE %----- FEATURE


% %%% Observe
% figure;subplot(3,1,1);plot(abs(fftshift(fftFiltPowX_I)),'r');
%        subplot(3,1,2);plot(abs(fftshift(fftFiltPowY_I)),'g');
%        subplot(3,1,3);plot(abs(fftshift(fftFiltPowZ_I)));
%        
% %%% Observe
% figure;plot(abs(fftshift(fftFiltPowX_I)),'r');hold on;
%        plot(abs(fftshift(fftFiltPowY_I)),'g');hold on;
%        plot(abs(fftshift(fftFiltPowZ_I)));hold off

%%% Get dominant frequencies
%%% Sum the FFT of all three channels of a sensor


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write to train file

% Assign labels
if i <= 20   % 1 to 52 idle samples =1
    label = 1;
elseif 21 <= i && i <= 33 %% ||  49 <= i && i <= 54 %% 53 to 78 ladder = 2
    label = 2;
elseif 34 <= i && i <= 87  %%  stone =3
    label = 3;
elseif 88 <= i && i <= 111  %% wire cut = 4
    label = 4;
end

FeatVec = [EnerX PowX MeanW_X MeanRE_X MeanFE_X PKSfftX' LOCSfftX'...
           EnerY PowY MeanW_Y MeanRE_Y MeanFE_Y PKSfftY' LOCSfftY'...
           EnerZ PowZ MeanW_Z MeanRE_Z MeanFE_Z PKSfftZ' LOCSfftZ'...
           label];

dlmwrite(['./test/' 'test.csv'],FeatVec,'delimiter',',','-append')
end
