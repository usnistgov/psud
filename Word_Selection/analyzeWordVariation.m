%Code to guide selection of MRT keywords to be used in for NIST MCV KPI
%measurements, S. Voran, 8/14/2020
%Basic approach is to minimize variation over trials. This should allow 
%the most stable and rapidly converging measurements.  Code selects 
%a specified number of words (equal number from each talker) or full
%batches of 6 words (equal number per talker)

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
%The 5 codecs are Analog FM, P25FR, P25HR, AMRNB7 (12.20 kbps), 
%AMRWB2 (12.65 kbps) and all are implemented in software only.
%
%The 3 intelligibility estimators are ABC-MRT16, STOI, and ESTOI.
%
%In this code we use the terms talker 1 to 4 for convenience.
%These map to the talker names used in the files as follows:
%Talker1 = Female 1
%Talker2 = Female 3
%Talker3 = Male 3
%Talker4 = Male 4

%take mean over trials, result is 1200 by 5 by 3
meanIntelScores = squeeze(mean(intelScores,2));

%take std over trials, result is 1200 by 5 by 3
stdIntelScores=squeeze(std(intelScores,[],2));

figure(1),plot(squeeze(mean(meanIntelScores)),'o-')
xlabel('Codec Number'),ylabel('Intel. Est.')
legend('ABC-MRT16','STOI','ESTOI','Location','SE'),grid
title('Grand Means (over 1200 Words and 120 Trials)')
disp('The 5 codecs are Analog FM, P25FR, P25HR, AMRNB7 (12.20 kbps),') 
disp('and AMRWB2 (12.65 kbps), in that order.')
disp(' ')
%Three estimators are in general agreement in their trends except 
%for Codec 1 (Analog FM).
%This report https://www.its.bldrdoc.gov/publications/details.aspx?pub=2490
%shows MRT scores for AFM are much higher than scores for P25FR.
%This agrees with ABC-MRT16, but does  not agree with STOI and ESTOI

figure(2),plot(squeeze(mean(stdIntelScores)),'o-')
xlabel('Codec Number'),ylabel('Std.')
legend('ABC-MRT16','STOI','ESTOI','Location','NE'),grid
title('Average (over 1200 Words) of Std (over 120 Trials)')
%ABC-MRT16 has higher average std than STOI and ESTOI in general, and 
%very much so for P25.  This is a consequence of the MRT word-choice
%paradigm used in ABC-MRT16, compared to the signal disturbance
%paradigm of STOI and ESTOI.  Yet we will find that we can locate
%talker+word combinations where ABC-MRT16 has std. that can complete
%with those of STOI and ESTOI.

%From here on we focus on std over trials for ABC-MRT16
%Work will be driven by the worst (greatest) std. over codecs.
%We want words that will be as useful as possible even in the worst case
%SUT which would be P25HR according to fig. 2 and intuition.

%--------------Finding best words---------------------

%compute 1200 by 1 vector with worst (over codecs) std (over trials) 
%for ABC-MRT16
worstStdIntelScores = max(stdIntelScores(:,:,1),[],2);
figure(3),plot(worstStdIntelScores,'o-')
xlabel('Word Number'),ylabel('Worst Std.'),grid
title('ABC-MRT16 Worst (aka largest) (over codecs) Std (over trials) for each word')

hold on %mark the 4 words we have been using: hook, west, cop, pay
plot([232 389 724 1006],worstStdIntelScores([232 389 724 1006]),'*r')
hold off
disp('On Fig. 3 the red stars are (L to R) F1 hook, F3 west, M3 cop, and M4 pay.')
disp(' ')
%Each block of 300 words in fig 3is from one talker.  Can see Talker 1 
%has lots of small (even zero) stds but Talker 3 has much fewer


disp('Number of words with zero std for each of the 4 talkers is:')
wordCounts = sum(reshape(worstStdIntelScores,300,4)==0)

nWords = 10; %number of words we seek from each talker

%Looking for nWords words from each talker.  Would like the words to be
%unique so we get maximum diversity when testing SUTs.  So we will pick
%the best unique words from each talker in order of inverse availability.
%First we pick words from Talker 3 (since it has the fewest options),
%then Talker 2, then Talker 4, then Talker 1. 

talker3 = worstStdIntelScores(601:900); %load talker 3 stds
[~,order] = sort(talker3); %sort them
%extract pointer to the best words, add talker offset to make it an absolute
%pointer into the list of 1200 words.
talker3Words = order(1:nWords) + 600; 
%record which of the 300 words we have used for talker 3
wordsUsed = order(1:nWords);

talker2 = worstStdIntelScores(301:600); %load talker 2 stds
[~,order] = sort(talker2); %sort them
order=setdiff(order,wordsUsed,'stable');%remove words already used
%but maintain order
%extract pointer to the best words, add talker offset to make it an absolute
%pointer into the list of 1200 words.
talker2Words = order(1:nWords) + 300; 
%append words used for talker 2 to the list of words used
wordsUsed = [wordsUsed; order(1:nWords)];

talker4 = worstStdIntelScores(901:1200); %load talker 4 stds
[~,order] = sort(talker4); %sort them
order=setdiff(order,wordsUsed,'stable');%remove words already used
%but maintain order
%extract pointer to the best words, add talker offset to make it an absolute
%pointer into the list of 1200 words.
talker4Words = order(1:nWords) + 900; 
%append words used for talker 4 to the list of words used
wordsUsed = [wordsUsed; order(1:nWords)];

talker1 = worstStdIntelScores(1:300); %load talker 1 stds
[~,order] = sort(talker1); %sort them
order=setdiff(order,wordsUsed,'stable');%remove words already used
%but maintain order
%extract pointer to the best words, add talker offset to make it an absolute
%pointer into the list of 1200 words.
talker1Words = order(1:nWords) + 0; 
%append words used for talker 1 to the list of words used
%size is nWords * 4 by 1.  Values range from 1 to 300, values do not
%indicate the talker is associated with each word
wordsUsed = [wordsUsed; order(1:nWords)];

%Size is nWords * 4 by 1, values range from 1 to 1200
absWordsUsed = [talker1Words;talker2Words;talker3Words;talker4Words];

%Extract worst (over codecs) std for STOI and ESTOI for comparison purposes
worstStdIntelScoresSTOI = max(stdIntelScores(:,:,2),[],2);
worstStdIntelScoresESTOI = max(stdIntelScores(:,:,3),[],2);
%form a matrix to display, nWords * 4 words selected by 3 intel. estimators
dispMat = [worstStdIntelScores(absWordsUsed), ...
    worstStdIntelScoresSTOI(absWordsUsed), ...
    worstStdIntelScoresESTOI(absWordsUsed)];

figure(4),plot(dispMat,'o-')
xlabel('Selected Word Number'),ylabel('Worst Std.')
legend('ABC-MRT16','STOI','ESTOI','Location','NW'),grid
title('Worst (over codecs) Std (over trials) for each selected word')
%When nWords = 12 we are able to get zero std for words except the
%final 5 words from talker 3.

figure(5),plot(squeeze(mean(meanIntelScores(absWordsUsed,:,:))),'o-')
xlabel('Codec Number'),ylabel('Intel. Est.')
legend('ABC-MRT16','STOI','ESTOI','Location','SE'),grid
title('Grand Means (over selected words and all 120 trials)')
%In MRT testing std is lowest at the extreme ends of the scale. The five
%codecs in this test produce results well above the center of the scale,
%so low std is associated with the top of the scale.  This means that
%picking words that give the lowest (often zero) ABC-MRT16 std is akin
%to picking words that give the highest (almost always 1.0) ABC-MRT16
%score.  So we have no ability to differentiate codes.  If the goal is
%to simply classify a radio link as usable or not usable, this should not
%be a problem. But if we were trying to estimate intelligibility, words 
%selected this way would create a large upward bias in results.

disp('Here are the files selected by the best words method.')
disp('If you want fewer, use the first ones listed for each talker')
disp('or change the value of nWords in this code. If you want more,')
disp('change the value of nWords in this code.')
selectedFilenames = fileNames(absWordsUsed)
%of course one can save selectedFilenames to a .mat or .txt file, or cut
%and paste from the screen

return
%--------------Finding best batches ---------------------
%A batch (or list) is a group of 6 words that are used together in an
%MRT trial

%form a 200 by 1 vector with worst (over codecs and words) std 
%(over trials) ABC-MRT16 scores
worstStdIntelScoresBatch = max(reshape(worstStdIntelScores,6,200))';
figure(6),plot(worstStdIntelScoresBatch,'o-')
xlabel('Batch Number'),ylabel('Worst Std.'),grid
title('ABC-MRT16 Worst (over codecs and words) Std (over trials) for each batch')
%Each block of 50 words is from one talker.  Can see Talker 1 has more 
%smaller stds and Talker 3 has very few.

nBatches = 2; %number of batches we seek from each talker

%Looking for nBatches batches from each talker.  Would like the batches
%to be unique so we get maximum diversity in testing SUTs.  So we will pick
%the best unique batches from each talker in order of inverse availability.
%First we pick batches from Talker 3 (since it has the fewest options),
%then Talker 2, then Talker 4, then Talker 1. 

talker3 = worstStdIntelScoresBatch(101:150); %load talker 3 stds
[~,order] = sort(talker3); %sort them
%extract pointer to the best batches, add talker offset to make it an absolute
%pointer into the list of 200 batches.
talker3Batches = order(1:nBatches) + 100; 
%record which of the 50 batches we have used for talker 3
batchesUsed = order(1:nBatches);

talker2 = worstStdIntelScoresBatch(51:100); %load talker 2 stds
[~,order] = sort(talker2); %sort them
order=setdiff(order,batchesUsed,'stable');%remove batches already used
%but maintain order
%extract pointer to the best batches, add talker offset to make it an absolute
%pointer into the list of 200 batches.
talker2Batches = order(1:nBatches) + 50; 
%append batches used for talker 2 to the list of batches used
batchesUsed = [batchesUsed; order(1:nBatches)];

talker4 = worstStdIntelScoresBatch(151:200); %load talker 4 stds
[~,order] = sort(talker4); %sort them
order=setdiff(order,batchesUsed,'stable');%remove batches already used
%but maintain order
%extract pointer to the best batches, add talker offset to make it an absolute
%pointer into the list of 200 batches.
talker4Batches = order(1:nBatches) + 150; 
%append batches used for talker 4 to the list of batches used
batchesUsed = [batchesUsed; order(1:nBatches)];

talker1 = worstStdIntelScoresBatch(1:50); %load talker 1 stds
[~,order] = sort(talker1); %sort them
order=setdiff(order,batchesUsed,'stable');%remove batches already used
%but maintain order
%extract pointer to the best batches, add talker offset to make it an absolute
%pointer into the list of 200 batches.
talker1Batches = order(1:nBatches) + 0; 
%append batches used for talker 1 to the list of batches used
%size is nBatches * 4 by 1. Values range from 1 to 50, values do not
%indicate which talker is associated with each batch
batchesUsed = [batchesUsed; order(1:nBatches)];

%Size is nBatches * 4 by 1, Range is 1 to 200
absBatchesUsed = [talker1Batches;talker2Batches;talker3Batches;talker4Batches];

%Extract worst (over codecs and words) std for STOI and ESTOI for
%comparison purposes
worstStdIntelScoresSTOIBatch = ...
    max(reshape(max(stdIntelScores(:,:,2),[],2),6,200))';
worstStdIntelScoresESTOIBatch = ...
    max(reshape(max(stdIntelScores(:,:,3),[],2),6,200))';
%form a matrix to display, nBatches * 4 batches selected by 3 intel. estimators
dispMat = [worstStdIntelScoresBatch(absBatchesUsed), ...
    worstStdIntelScoresSTOIBatch(absBatchesUsed), ...
    worstStdIntelScoresESTOIBatch(absBatchesUsed)];

figure(7),plot(dispMat,'o-')
xlabel('Selected Batch Number'),ylabel('Worst Std.')
legend('ABC-MRT16','STOI','ESTOI','Location','NW'),grid
title('Worst (over codecs and words) Std (over trials) for each Selected Batch')
%ABC-MRT16 stds are larger than those of STOI and ESTOI.
%These ABC-MRT16 stds are mostly in the same range as those of the original
%4 words (hook, west, cop, pay have 0.041, 0.062, 0.024, and 0.095 std resp.)
%T3 is the exception and has greater stds

%Extract mean(over words) of worst (over codecs) of std (over trials) 
%ABC-MRT16, STOI, and ESTOI to get an alternate view
M1 = mean(reshape(max(stdIntelScores(:,:,1),[],2),6,200))';
M2 = mean(reshape(max(stdIntelScores(:,:,2),[],2),6,200))';
M3 = mean(reshape(max(stdIntelScores(:,:,3),[],2),6,200))';

%form a matrix to display, nBatches * 4 choosen batches, by 3 intel. estimators
dispMat = [M1(absBatchesUsed),M2(absBatchesUsed),M3(absBatchesUsed)];
figure(8),plot(dispMat,'o-')
xlabel('Selected Batch Number'),ylabel('Worst Std.')
legend('ABC-MRT16','STOI','ESTOI','Location','NW'),grid
title('Mean (over words) of Worst (over codecs) Std (over trials) for each Selected Batch')

%Convert batches to words
wordsOfBatches = (absBatchesUsed-1)*6 + [1:6];
wordsOfBatches = reshape(wordsOfBatches',4*nBatches*6,1);

disp('Here are the files selected by the best batches method.')
disp('If you want more or fewer change the value of nBatches in this code.')
selectedFilenames = fileNames(wordsOfBatches)
%of course one can save selectedFilenames to a .mat or .txt file, or cut
%and paste from the screen

disp('Words that are selected by both word method and batch method are')
commonWords = intersect(wordsOfBatches,absWordsUsed);
selectedFilenames = fileNames(commonWords)
%With nWords = 12 and nBatches = 2, we get 6 common words and they are
%2 words from a talker2 batch, 2 words from a talker4 batch, and
%one word from each of the talker3 batches

figure(9),plot(squeeze(mean(meanIntelScores(wordsOfBatches,:,:))),'o-')
xlabel('Codec Number'),ylabel('Intel. Est.')
legend('ABC-MRT16','STOI','ESTOI','Location','SE'),grid
title('Grand Means (over selected batches and all 120 trials)')
%In MRT testing std is lowest at the extreme ends of the scale. The five
%codecs in this test produce results well above the center of the scale,
%so low std is associated with the top of the scale.  This means that
%picking batches that give the lowest ABC-MRT16 std is akin to picking
%batches that give the highest ABC-MRT16 score.  So we get higher means
%than in figure 1 where all 1200 words are used.  But we get lower 
%means than in figure 5 where we were picking best words.  Best words is
%a finer-grained optimization than best batches.