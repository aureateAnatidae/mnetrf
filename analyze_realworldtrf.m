close all; clear all; clc;
init_unfold;
dataFolder = './data';
dataFile = 'May20lsl_synchronize_test_02.xdf';

%% Load and preprocess the EEG
EEG = eeg_load_xdf(fullfile(dataFolder,dataFile));
EEG = pop_select(EEG,'nochannel',5); % Remove unused aux channel
EEG = pop_eegfiltnew(EEG, 0.1, 15);

%% Load the streams
streams = load_xdf(fullfile(dataFolder,dataFile));
eegStream = streams{1};
eegTimes = eegStream.time_stamps;
audioStream = streams{2};
audio = audioStream.time_series;
audio = rescale(audio,-1,1);
audioTimes = audioStream.time_stamps;
audioSR = str2num(audioStream.info.nominal_srate);
% plot(audioTimes,audio');

%% Envelope
htAudio = hilbert(audio);
envAudio = abs(htAudio);
fEnvAudio = lowpass(envAudio,128,audioSR);
fEnvAudio = mean(fEnvAudio,1);
fEnvAudio = rescale(fEnvAudio,0,1);
dsFEnvAudio = interp1(audioTimes,fEnvAudio,eegTimes);
% plot(eegTimes,dsFEnvAudio);

%% TRF
uEEG = EEG;
for j = 1:4
    uEEG.chanlocs(j).type = [];
end

% Add an event to EEG
uEEG.event(1).type = '1';
uEEG.event(1).latency = 1;
uEEG = eeg_checkset(uEEG);

% Flag bad segments of data
winrej = uf_continuousArtifactDetect(uEEG,'amplitudeThreshold',250,'windowsize',2000,'stepsize',1000,'combineSegments',[]);

% Time window for TRF
ufTime = [-0.5 1];

% Add audio signal to data
uEEG.data(5,:) = dsFEnvAudio;

% Generate design matrix
uEEG = uf_timeexpandDesignmat_addTRF(uEEG,'channel',5,'name','envelope','timelimits',ufTime);

% Remove audio channel and do artifact rejection
uEEG.data(5,:) = [];
uEEG = uf_continuousArtifactExclude(uEEG,struct('winrej',winrej));

% Grab the data and the design matrix
data = double(uEEG.data);
X = uEEG.unfold.Xdc;

% Flag zero rows
nonZero = any(X,2);
isZero = ~nonZero;
data(:,isZero') = [];
X(isZero',:) = [];

% Flag and remove any rows with NaN
isNan = isnan(sum(X,2));
data(:,isNan) = [];
X(isNan,:) = [];

%% Do cross-validation (optional)
% regtype = 'onediff';
% lambdas = [1E2 1E3 1E4 1E5 1E6 1E7 1E8 1E9 1E10];
% k = 10;
% [allErrors,bestBeta] = doRegCV(data,X,regtype,{1:size(X,2)},{[]},lambdas,k);
% figure();
% plot(allErrors);
% [~,j] = min(allErrors);
% disp(lambdas(j));

%% Solve GLM
regtype = 'onediff';
lambda = 1E5; % Get this from cross-validation above
thisPDM = pinv_reg(X,lambda,regtype);
tempBeta = thisPDM * data';

%%
figure();
plot(uEEG.unfold.times,tempBeta);
legend({uEEG.chanlocs.labels});
