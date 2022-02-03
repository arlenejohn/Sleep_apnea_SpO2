list_string={'ucddb002','ucddb003','ucddb005','ucddb006','ucddb007','ucddb008','ucddb009',...
             'ucddb010','ucddb011','ucddb012','ucddb013','ucddb014','ucddb015','ucddb017',...
             'ucddb018','ucddb019','ucddb020','ucddb021','ucddb022','ucddb023','ucddb024',...
             'ucddb025','ucddb026','ucddb027','ucddb028'};
df_minority_list=0;
for l=1:length(list_string)

    load(strcat(list_string{l},'.mat'));
    ecg=signal(:,1);
    ecg=(ecg-mean(ecg))/std(ecg);
    spo2=signal(:,2);
    %spo2=(spo2-mean(spo2))/std(spo2);
    sum_mov=signal(:,3);
    sum_mov=(sum_mov-mean(sum_mov))/std(sum_mov);
    ribcage=signal(:,4);
    ribcage=(ribcage-mean(ribcage))/std(ribcage);
    abdo=signal(:,5);
    abdo=(abdo-mean(abdo))/std(abdo);
    load(strcat(list_string{l},'_label.mat'));
    labels=labels(1:length(labels)-11);
    
    ecg_features=zeros(round(length(ecg)/128)-11,1408);
    i = 129;
    k=1;
    while i < length(ecg)-1279
        feature_window=ecg(i-128: i+1279);
        feature_window=feature_window';
        ecg_features(k,:)=feature_window;
        k=k+1;
        i=i+128;
    end
    
    spo2_features=zeros(round(length(spo2)/128)-11,1408);
    i = 129;
    k=1;
    while i < length(spo2)-1279
        feature_window=spo2(i-128: i+1279);
        feature_window=feature_window';
        spo2_features(k,:)=feature_window;
        k=k+1;
        i=i+128;
    end
    
    sum_mov_features=zeros(round(length(sum_mov)/128)-11,1408);
    i = 129;
    k=1;
    while i < length(sum_mov)-1279
        feature_window=sum_mov(i-128: i+1279);
        feature_window=feature_window';
        sum_mov_features(k,:)=feature_window;
        k=k+1;
        i=i+128;
    end
    
    ribcage_features=zeros(round(length(ribcage)/128)-11,1408);
    i = 129;
    k=1;
    while i < length(ribcage)-1279
        feature_window=ribcage(i-128: i+1279);
        feature_window=feature_window';
        ribcage_features(k,:)=feature_window;
        k=k+1;
        i=i+128;
    end
    
    abdo_features=zeros(round(length(abdo)/128)-11,1408);
    i = 129;
    k=1;
    while i < length(abdo)-1279
        feature_window=abdo(i-128: i+1279);
        feature_window=feature_window';
        abdo_features(k,:)=feature_window;
        k=k+1;
        i=i+128;
    end
    
     
    clear ecg
    clear spo2
    clear ribcage
    clear sum_mov
    clear abdo
    clear signal
    
    total_features=[ecg_features spo2_features sum_mov_features ribcage_features abdo_features];
    spo2_rows=total_features(:,1409:2816);
    idx=[];
    for k=1:length(spo2_rows)
        check_val=spo2_rows(k,:);
        if find(check_val<0.5)
            idx=[idx;k];
        end
    end
    write_val=[];
    labe=[];
    for k=1:length(spo2_rows)
        if ~ismember(k,idx)
            write_val=[write_val;total_features(k,:)];
            labe=[labe;labels(k,:)];
        end
    end
    labels=labe;
    total_features=write_val;
    spo2_rows=(total_features(:,1409:2816)-mean(total_features(:,1409:2816)))/std(total_features(:,1409:2816));
    total_features(:,1409:2816)=spo2_rows;
        
    clear ecg_features
    clear spo2_features
    clear sum_mov_features
    clear ribcage_features
    clear abdo_features
    clear spo2_rows
    clear labe
    clear write_val
    clear idx
    
    c = cvpartition(labels,'HoldOut',0.1);
    idxTrain=training(c);
    idxTest=test(c);
    test_list=total_features(idxTest,:);
    class_test=labels(idxTest);
    save(fullfile(folder,strcat(list_string{l},'_test_labels.mat')),'class_test');
    clear class_test
    ecg_test=test_list(:,1:1408);
    save(fullfile(folder,(strcat(list_string{l},'_ecg_test.mat')),'ecg_test');
    clear ecg_test
    spo2_test=test_list(:,1409:2816);
    save(fullfile(folder,(strcat(list_string{l},'_spo2_test.mat')),'spo2_test');
    clear spo2_test
    sum_mov_test=test_list(:,2817:4224);
    save(fullfile(folder,(strcat(list_string{l},'_sum_mov_test.mat')),'sum_mov_test');
    clear sum_mov_test
    ribcage_test=test_list(:,4225:5632);
    save(fullfile(folder,(strcat(list_string{l},'_ribcage_test.mat')),'ribcage_test');
    clear ribcage_test
    abdo_test=test_list(:,5633:7040);
    save(fullfile(folder,(strcat(list_string{l},'_abdo_test.mat')),'abdo_test');
    clear abdo_test
    clear test_list
    
    train_valid_list=total_features(idxTrain,:);
    class_train_valid=labels(idxTrain);
    
    clear total_features
    clear labels
    
    c = cvpartition(class_train_valid,'HoldOut',0.1/0.9);
    idxTrain=training(c);
    idxValid=test(c);
    
    
    train_list=train_valid_list(idxTrain,:);
    class_train=class_train_valid(idxTrain);
    
    
    df_majority = train_list(class_train==0,:);
    df_minority = train_list(class_train==1,:);
    
    if isempty(df_minority)
        s = RandStream('mlfg6331_64');
        y = datasample(s,1:size(df_minority_list,1),size(df_majority,1),'Replace',true);
        df_minority_upsampled = df_minority_list(y,:);
        total_features_resampled = [df_majority;df_minority_upsampled];
        labels_resampled=[zeros(size(df_majority,1),1);ones(size(df_minority_upsampled,1),1)];
        
    else
        s = RandStream('mlfg6331_64');
        y = datasample(s,1:size(df_minority,1),size(df_majority,1)-size(df_minority,1),'Replace',true);
        df_minority_upsampled = df_minority(y,:);
        total_features_resampled = [df_majority;df_minority; df_minority_upsampled];
        labels_resampled=[zeros(size(df_majority,1),1);ones(size(df_minority,1),1);ones(size(df_minority_upsampled,1),1)];
        if size(df_minority,1)>size(df_minority_list,1)
            df_minority_list=df_minority;
        end
    end
    class_train=labels_resampled;
    train_list=total_features_resampled;
    
    clear y
    clear s
    clear labels_resampled
    clear total_features_resampled
    
    
    save(strcat(list_string{l},'_train_labels.mat'),'class_train');
    clear class_train
    ecg_train=train_list(:,1:1408);
    save(strcat(list_string{l},'_ecg_train.mat'),'ecg_train');
    clear ecg_train
    spo2_train=train_list(:,1409:2816);
    save(strcat(list_string{l},'_spo2_train.mat'),'spo2_train');
    clear spo2_train
    sum_mov_train=train_list(:,2817:4224);
    save(strcat(list_string{l},'_sum_mov_train.mat'),'sum_mov_train');
    clear sum_mov_train
    ribcage_train=train_list(:,4225:5632);
    save(strcat(list_string{l},'_ribcage_train.mat'),'ribcage_train');
    clear ribcage_train
    abdo_train=train_list(:,5633:7040);
    save(strcat(list_string{l},'_abdo_train.mat'),'abdo_train');
    clear abdo_train
    clear train_list
    clear idxTrain
    
    
    valid_list=train_valid_list(idxValid,:);
    class_valid=class_train_valid(idxValid);
    df_majority = valid_list(class_valid==0,:);
    df_minority = valid_list(class_valid==1,:);
    
    if isempty(df_minority)
        s = RandStream('mlfg6331_64');
        y = datasample(s,1:size(df_minority_list,1),size(df_majority,1),'Replace',true);
        df_minority_upsampled = df_minority_list(y,:);
        total_features_resampled = [df_majority;df_minority_upsampled];
        labels_resampled=[zeros(size(df_majority,1),1);ones(size(df_minority_upsampled,1),1)];
        
    else
        s = RandStream('mlfg6331_64');
        y = datasample(s,1:size(df_minority,1),size(df_majority,1)-size(df_minority,1),'Replace',true);
        df_minority_upsampled = df_minority(y,:);
        total_features_resampled = [df_majority;df_minority; df_minority_upsampled];
        labels_resampled=[zeros(size(df_majority,1),1);ones(size(df_minority,1),1);ones(size(df_minority_upsampled,1),1)];
        if size(df_minority,1)>size(df_minority_list,1)
            df_minority_list=df_minority;
        end
    end
    class_valid=labels_resampled;
    valid_list=total_features_resampled;
    clear labels_resampled
    clear total_features_resampled
    
    
    save(strcat(list_string{l},'_valid_labels.mat'),'class_valid');
    clear class_valid
    ecg_valid=valid_list(:,1:1408);
    save(strcat(list_string{l},'_ecg_valid.mat'),'ecg_valid');
    clear ecg_valid
    spo2_valid=valid_list(:,1409:2816);
    save(strcat(list_string{l},'_spo2_valid.mat'),'spo2_valid');
    clear spo2_valid
    sum_mov_valid=valid_list(:,2817:4224);
    save(strcat(list_string{l},'_sum_mov_valid.mat'),'sum_mov_valid');
    clear sum_mov_valid
    ribcage_valid=valid_list(:,4225:5632);
    save(strcat(list_string{l},'_ribcage_valid.mat'),'ribcage_valid');
    clear ribcage_valid
    abdo_valid=valid_list(:,5633:7040);
    save(strcat(list_string{l},'_abdo_valid.mat'),'abdo_valid');
    clear abdo_valid
    clear valid_list
    clear idxValid
    
    clearvars -except list_string l df_minority_list
    
    list_string{l}
end
    
