load wordVariation

%loads in two variables: intelScores and fileNames
%
%intelScores is 1200 words by 120 trials by 5 codecs by 3
%intelligibility estimators
%
%The 1200 words are in standard order and this order is given in the
%variable fileNames.  The 1200 is formed from 6 words per batch x
%50 batches per talker x 4 talkers.
%
%The 120 trials are due to different white noise realizations and speech-
%codec timing relationships. The noise was added to the speech at 20 dB SNR
%before transmission.  This level is clearly audible but does not impair
%intelligibility.
%


%In this code we use the terms talker 1 to 4 for convenience.
%These map to the talker names used in the files as follows:
%Talker1 = Female 1
%Talker2 = Female 3
%Talker3 = Male 3
%Talker4 = Male 4

%The 5 codecs are Analog FM, P25FR, P25HR, AMRNB7 (12.20 kbps),
%AMRWB2 (12.65 kbps) and all are implemented in software only.
codecs = {'Analog FM', 'P25FR', 'P25HR', 'AMRNB7','AMRWB2'};

%The 3 intelligibility estimators are ABC-MRT16, STOI, and ESTOI.
estimators = {'ABC-MRT16','STOI','ESTOI'};

% Preallocate cell array to store data
data_out = cell(numel(intelScores)+1,ndims(intelScores)+1);

% Define header
data_out(1,:) = {'Clip','Codec','Estimator','Trial','Value'};
% Count variable
count = 2;

for clip = 1:size(intelScores,1)
    for codec = 1:size(intelScores,3)
        for trial = 1:size(intelScores,2)
            for estimator = 1:size(intelScores,4)
                dat_val = intelScores(clip,trial,codec,estimator);
                data_out(count,:) = {fileNames{clip};
                    codecs{codec};
                    estimators{estimator};
                    trial;
                    dat_val};
                count = count + 1;
            end
        end
    end
end

writecell(data_out,'intelScores.csv');