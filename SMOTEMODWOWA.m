function [Xover,Yover] = SMOTEMODWOWA(X,Y,k,over_param,norm_type,fs_type,nvars,alphaOWA,quantif)
%step 1: identify minority class Y={-1,+1} format
fin1=find(Y==1); 
fin2=find(Y==-1);
Nej1=size(fin1,1);
Nej2=size(fin2,1);
if Nej1<Nej2
    Xmin=X(fin1,:); 
else
    Xmin=X(fin2,:); 
end  
%step 2: Fisher Score, select "nvars" variables
switch fs_type
    case 1
        weights=zeros(size(X,2),1);
        pos=find(Y==1);
        neg=find(Y==-1);
        for ii = 1:size(X,2),
            weights(ii)= (abs(mean(X(pos,ii))-mean(X(neg,ii))))/(var(X(pos,ii))+var(X(neg,ii)));
        end
    case 2
        for ii = 1:size(X,2),
            weights(ii)= max(muteinf(X(:,ii),Y),0);
        end
        weights=weights';
    case 3
        X_train=X;
        Y_train=Y;
        % Preprocessing
        s_n = X_train(Y_train==-1,:);
        s_p = X_train(Y_train==1,:);
        mu_sn = mean(s_n);
        mu_sp = mean(s_p);
        % Metric 1: Mutual Information
        mi_s = [];
        for i = 1:size(X_train,2)
            mi_s = [mi_s, muteinf(X_train(:,i),Y_train)];
        end       
        % Metric 2: class separation
        sep_scores = ([mu_sp - mu_sn].^2);
        st   = std(s_p).^2;
        st   = st+std(s_n).^2;
        f=find(st==0); %% remove ones where nothing occurs
        st(f)=10000;  %% remove ones where nothing occurs
        sep_scores = sep_scores ./ st;        
        % Building the graph
        vec = abs(sep_scores + mi_s )/2;
        % Building the graph
        Kernel_ij = [vec'*vec] ;      
        Kernel_ij = Kernel_ij - min(min( Kernel_ij ));
        Kernel_ij = Kernel_ij./max(max( Kernel_ij ));     
        % Standard Deviation
        STD = std(X_train,[],1);
        STDMatrix = bsxfun( @max, STD, STD' );
        STDMatrix = STDMatrix - min(min( STDMatrix ));
        sigma_ij = STDMatrix./max(max( STDMatrix ));     
        Kernel =  (0.5*Kernel_ij+(1-0.5)*sigma_ij);    
        % Eigenvector Centrality and Ranking
        [eigVect, ~] = eigs(double(Kernel),1,'lm');
        weights=abs(eigVect);  
    case 4
        corrMatrix = abs( corr(X) );
        % Ranking according to minimum correlations
        weights = min(corrMatrix,[],2);
    otherwise
        weights=zeros(size(X,2),1);
        pos=find(Y==1);
        neg=find(Y==-1);
        for ii = 1:size(X,2),
            weights(ii)= (abs(mean(X(pos,ii))-mean(X(neg,ii))))/(var(X(pos,ii))+var(X(neg,ii)));
        end
end
%step 3: modified SMOTE, "norm_type" norm using only selected attributes
%puede estar al reves la importancia (más irrelevantes al comienzo, asumo eso)
T_dagger=Xmin;
[v1,rank]=sort(weights,'descend');
XminSel=Xmin(:,rank(1:nvars));
W = OWAFunction(size(XminSel,2),alphaOWA,quantif); %ya ordenado por relevancia
%W=sqrt(v1(1:nvars)/mean(v1(1:nvars)))';
D=pdist(repmat(W,size(XminSel,1),1).*XminSel,norm_type);
D=squareform(D); %square matrix instead of a vector, easier for selecting k neighbors
for i=1:size(Xmin,1)
    [rankDist PosDist]=sort(D(i,:));
    r = randi([1 k],1,over_param); %"over_param" random integer numbers between 1 and "k"
    %r=[1:over_param]; %non-random version, "over_param" nearest neighbors
    %u=rand; %number "u" between 0 and 1
    u=0.3; %non-random version 
    for i2=1:over_param
        index=PosDist(r(i2)+1); %the +1 is required to avoid selecting the same obs. as neighbor
        new_P=(1-u).*Xmin(i,:)+u.*Xmin(index,:);
        T_dagger = [T_dagger;new_P];
    end 
end
%step 4: reconstruct (X,Y)
if Nej1<Nej2
    Xover=[T_dagger;X(fin2,:)];
    Yover=[ones(size(T_dagger,1),1);Y(fin2)];
else
    Xover=[T_dagger;X(fin1,:)];
    Yover=[-ones(size(T_dagger,1),1);Y(fin1)];
end  
