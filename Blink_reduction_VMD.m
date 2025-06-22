%%
close all 
clear
load sub10.mat

fs = 200;  

chunk_num = 1;
chunk_len = fs*27;


datain = sim10_con(1,1:chunk_len);
datain_GT = sim10_resampled(1,1:chunk_len);

%% --0 SETTINGS and hyperparameters----------------------------------------
fs           = 200;        % 
K            = 16;          % number of IMFs to extract
penaltyFac   = 2000;       % bandwidth penalty
padSeconds   = 0.5;        % mirror-pad 0.5 s at each edge

%% --1. MIRROR-PAD THE DATA -------------------------------
padSamples = round(padSeconds*fs);
eegRaw   = datain(:).';       
eegGT   = datain_GT(:).';   

eegPad   = [fliplr(eegRaw(1:padSamples))   eegRaw   fliplr(eegRaw(end-padSamples+1:end))];
eegGTPad = [fliplr(eegGT(1:padSamples)) eegGT fliplr(eegGT(end-padSamples+1:end))];


%% -- 2. VARIATIONAL MODE DECOMPOSITION --------------------
% Name-value syntax: 'NumIMFs' sets K, 'PenaltyFactor' is α :contentReference[oaicite:0]{index=0}
imfEEG = vmd(eegPad, ...
             NumIMFs        = K, ...
             PenaltyFactor  = penaltyFac, ...
             InitializeMethod = "random" ...
             );
imfEEGGT = vmd(eegGTPad, ...
             NumIMFs        = K, ...
             PenaltyFactor  = penaltyFac, ...
             InitializeMethod = "random" ...
             );

% Trim off the mirror padding
imfEEG   = (imfEEG(padSamples+1:end-padSamples,:));
imfEEG   = imfEEG';

imfEEGGT   = (imfEEGGT(padSamples+1:end-padSamples,:));
imfEEGGT   = imfEEGGT';


%% ---------------- 3A. TIME-DOMAIN PLOTS --------------------------------
t  = (0:numel(eegRaw)-1)/fs;
figure('Name','VMD Modes – Time Domain','Color','w');
tiledlayout(K,2,'TileSpacing','compact','Padding','compact');

for k = 1:K
    nexttile;  plot(t, imfEEG(k,:));  axis tight;
    ylabel(sprintf('IMF %d',k)); if k==1, title('Contaminated EEG (FP1)'); end
    if k==K,   xlabel('Time (s)'); end

    nexttile;  plot(t, imfEEGGT(k,:));  axis tight;
    ylabel(sprintf('IMF %d',k)); if k==1, title('Ground-Truth EEG (FP1)'); end
    if k==K,   xlabel('Time (s)'); end
end
%% ---------------- 3B. POWER-SPECTRUM PLOTS -----------------------------
figure('Name','VMD Modes – Welch PSD','Color','w');
rows = ceil(K/2); tiledlayout(K,2,'TileSpacing','compact','Padding','compact');
kurPSD = zeros (1,k);
kurPSD_GT = zeros (1,k);
for k = 1:K
    % EEG IMF
    [Pxx,f] = pwelch(imfEEG(k,:),hamming(2*fs),[],[],fs);
    kurPSD(k) = kurtosis (10*log10(Pxx));
    nexttile;  plot(f,10*log10(Pxx)); xlim([0 fs/2]); grid on;
    ylabel(sprintf('IMF %d',k)); if k==1, title('Contaminated EEG PSD (FP1)'); end
    if k> K-2, xlabel('Hz'); end
    
    % EEGGT IMF
    [Pxx,f] = pwelch(imfEEGGT(k,:),hamming(2*fs),[],[],fs);
    kurPSD_GT(k) = kurtosis (10*log10(Pxx));
    nexttile;  plot(f,10*log10(Pxx)); xlim([0 fs/2]); grid on;
    ylabel(sprintf('IMF %d',k)); if k==1, title('Ground-Truth EEG PSD (FP1)'); end

    if k> K-2, xlabel('Hz'); end
end


%% -- 4a. Autocorrelation -------------------------------------------------
sigLen    = size(imfEEG,2);
Autocors = zeros(k,(sigLen*2) -1);
Autocors_GT = zeros(k,(sigLen*2) -1);
for i= 1:k
Autocors(i,:) = xcorr(imfEEG(i,:),imfEEG(i,:));
Autocors_GT(i,:) = xcorr(imfEEGGT(i,:),imfEEGGT(i,:));
end
%% -- 4b. AUTOcors PLOTS -------------------------------------------------
figure('Name','Autocorrelation','Color','w');
K= 16;
tiledlayout(K,2,'TileSpacing','compact','Padding','compact');
t_autocors        = (-0.5 * size(Autocors,2):0.5 * size(Autocors,2)-1)/fs;
for k = 1:K
    nexttile;  plot(t_autocors, Autocors(k,:));  axis tight;
    ylabel(sprintf('IMF %d',k)); if k==1, title('Contaminated EEG (FP1)'); end
    if k==K,   xlabel('Time (s)'); end

    nexttile;  plot(t_autocors, Autocors_GT(k,:));  axis tight;
    ylabel(sprintf('IMF %d',k)); if k==1, title('Ground-Truth EEG (FP1)'); end
    if k==K,   xlabel('Time (s)'); end
end

%% -- 5a. Kurtosis -------------------------------------------------------
kur = kurtosis(Autocors');
kur_scaled  = kur /max(kur);

%% -- 5b. Apply shifted logistic weight activation -----------------------

x= kur_scaled;
c= 0.3;
g=5;
num = (x./c).^g;
den = num + ((1 - x)./(1 - c)).^g;
y = num ./ den

kur_scaled_processed =y;

%% -- 6. reconstruction --------------------------------------------------

mixing_vec = kur_scaled_processed;
mixing_mat =  mixing_vec .* eye(length(mixing_vec));
cleanEEG = sum(mixing_mat * imfEEG,1);


%% -- 6. plotting filtered signal ----------------------------------------
t  = (0:numel(eegRaw)-1)/fs;
figure 

plot(t,eegGT,'k', LineWidth=1);
hold on
plot(t,eegRaw,'r', LineWidth=1);
plot(t,cleanEEG,'b', LineWidth=1);

xlabel("seconds")
ylabel("uV")

legend("Groundtruth","Contaminated","Filtered")
% xlim([1 6])

%% mean absolute errors

absErr_raw = abs(eegGT(:)-eegRaw(:));
mae_raw = mean(absErr_raw);

absErr = abs(eegGT(:)-cleanEEG(:));
mae = mean(absErr);


mae_vec = [mae_raw mae]

fprintf('  mae raw  : %.2f\n', mae_raw);
fprintf('  mae filtered  : %.2f\n', mae);