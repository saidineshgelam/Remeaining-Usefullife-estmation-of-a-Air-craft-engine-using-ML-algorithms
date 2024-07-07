%% Top Matter
clear;clc; close;
%%
format long; format compact;
set(0,'defaultTextInterpreter','latex'); %trying to set the default

sz = 60; %Marker Size
szz = sz/35;
lw = 10;
ms=8;
fs=32;
txtsz = 30;
txtFactor = 0.8;
ax = [0.9,1.4,0.0,2.0];
loc = 'southwest';
pos = [218,114,1478,796];
% txtsz = 24;
%% Load in data
trainingData = load("Training_DataSet\TrainingData_Engine.mat");
testingData =load("Testing_DataSet\TestingData_Engine.mat");

dataAll = trainingData.Train_EngineRun;
varName = trainingData.Variable_List;

dataAllTesting = testingData.Testing_EngineRun;
varNameTesting = testingData.Variable_List;

%% set up training data
dataArray = ones(1,25);

for i=1:length(dataAll)
    tempStruct = dataAll{i};
    tempArr = tempStruct.Data;
    for j=1:length(tempArr(:,1))
        dataArray(end+1,1:24) = tempArr(j,:);
        dataArray(end,25) = j-length(tempArr(:,1));         

    end
    clear temp tempStruct tempArr
end
dataArray(1,:) = [];

%% Set up testing data
dataArrayTest = ones(1,25);

for i=1:length(dataAllTesting)
    tempStruct = dataAllTesting{i};
    tempArr = tempStruct.Data;
    for j=1:length(tempArr(:,1))
        dataArrayTest(end+1,1:24) = tempArr(j,:);
        % dataArray(end,25) = j/length(tempArr(:,1));
         index=length(tempArr(:,1))-j;
         if(index>130)
             index=130;
         end
         dataArray(end,25) = index/130;    
    end
    clear temp tempStruct tempArr
end
dataArrayTest(1,:) = [];


%% Train RNN
rng(1234)
clear Input Output;
Input = (dataArray(:,1:24)');
Output = (dataArray(:,25)');


inputs = tonndata(Input,true,false);
target = tonndata(Output,true,false);

RNN_15x1_V01_12_V01 = layrecnet(1:2,[15]);
RNN_15x1_V01_12_V01.trainParam.epochs = 500;

[Xs,Xi,Ai,Ts] = preparets(RNN_15x1_V01_12_V01,inputs,target);
RNN_15x1_V01_12_V01 = train(RNN_15x1_V01_12_V01,Xs,Ts,Xi,Ai);



%% Calculate RUL for testing data

for i=1:100
        sampleID = i;
    
        tempStruct = dataAllTesting{sampleID};
        tempArr = tempStruct.Data';
        tempCell = tonndata(tempArr,true,false);
        tempOut = cell2mat(RNN_15x1_V01(tempCell));
        cellOut{i} = tempOut;
        predArray(i,1) = i;
        predArray(i,2) = length(tempOut);
        predArray(i,3:7) = tempOut(end-4:end);
  
end

