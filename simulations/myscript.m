%% Simulation of Ariana's experiment.

% seed=1; scaling_factor=1;
function[] = myscript(seed,scaling_factor)

%% Define variables that repeat over trials.

start_time = tic;

rng(seed);

% Number of neurons in each population
N = 10000;
Ne = 0.8*N;
Ni = 0.2*N;

% Number of neurons in ffwd layer
Nx=0.2*N;

% Recurrent net connection probabilities
P=[0.1 0.1; 0.1 0.1];

% Ffwd connection probs
Px=[.1; .1];

% Correlation between the spike trains in the ffwd layer
c=0.1;

% Timescale of correlation
taujitter=5;

% Mean connection strengths between each cell type pair
Jm=scaling_factor*[10 -100; 112.5 -250]/sqrt(N); % Changed old jei to steady state weight jei (-100).
Jxm=[180; 135]/sqrt(N);

% Time discretization
dt=.1;

% Proportions
qe=Ne/N;
qi=Ni/N;
qf=Nx/N;

% FFwd spike train rate (in kHz)
rx=10/1000;

% Build mean field matrices
Q=[qe qi; qe qi];
Qf=[qf; qf];
W=P.*(Jm*sqrt(N)).*Q;
Wx=Px.*(Jxm*sqrt(N)).*Qf;

% Synaptic timescales
taux=10;
taue=8;
taui=4;

% Generate connectivity matrices
tic
J=[Jm(1,1)*binornd(1,P(1,1),Ne,Ne) Jm(1,2)*binornd(1,P(1,2),Ne,Ni); ...
   Jm(2,1)*binornd(1,P(2,1),Ni,Ne) Jm(2,2)*binornd(1,P(2,2),Ni,Ni)];
Jx=[Jxm(1)*binornd(1,Px(1),Ne,Nx); Jxm(2)*binornd(1,Px(2),Ni,Nx)];
tGen=toc;
disp(sprintf('\nTime to generate connections: %.2f sec',tGen))
initialJ=J;
% Neuron parameters
Cm=1;
gL=1/15;
EL=-72;
Vth=-50;
Vre=-75;
DeltaT=1;
VT=-55;


%%%%% Plasticity params %%%%%%
tauSTDP=200;
% EE synapse
Jmax_ee = 10/sqrt(N);
eta_ee=0/1000; % Learning rate of EE

% EI synapse
Jmax_ei = -200/sqrt(N);
eta_ei=0.00075/Jmax_ei; % Learning rate of EI
rho0=0.010; % Target rate for e cells
alpha_e=2*rho0*tauSTDP;

% IE synapse - note that this can be hebbian or homeostatic
Jmax_ie = 200/sqrt(N);
eta_ie_homeo=0.000/Jmax_ie; % Learning rate of EI
rho0=0.023; % Target rate for i cells.
alpha_ie=2*rho0*tauSTDP;
% IE Hebbian
eta_ie_hebbian = 0/1000;
Jmax_ie = 112.5/sqrt(N); % only if hebbian

% II synapse
Jmax_ii = -200/sqrt(N);
eta_ii=0.000/Jmax_ii; % Learning rate of EI
rho0=0.023; % Target rate for i cells.
alpha_ii=2*rho0*tauSTDP;

% Indices of neurons to record currents, voltages
nrecord0=10; % Number to record from each population
Irecord=[randperm(Ne,nrecord0) randperm(Ni,nrecord0)+Ne];
numrecord=numel(Irecord); % total number to record

% Synaptic weights to record.
% The first row of Jrecord is the postsynaptic indices
% The second row is the presynaptic indices
nJrecord0=100; % Number to record
[II,JJ]=find(J(1:Ne,Ne+1:N)); % Find non-zero I to E weights
III=randperm(numel(II),nJrecord0); % Choose some at random to record
II=II(III);
JJ=JJ(III);
Jrecord_ei=[II JJ+Ne]'; % Record these
numrecordJ_ei=size(Jrecord_ei,2);
if(size(Jrecord_ei,1)~=2)
    error('Jrecord must be 2xnumrecordJ');
end

% Integer division function
IntDivide=@(n,k)(floor((n-1)/k)+1);


%% %% Burn in period to let the network reach steady state.%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Tburn = 500000; % burn in period total time (in ms).
timeBP = dt:dt:Tburn; 
% Number of time bins to average over when recording
nBinsRecord=20;
dtRecord=nBinsRecord*dt;
timeRecord=dtRecord:dtRecord:Tburn;
Ntrec=numel(timeRecord);
Nt=round(Tburn/dt);

% Maximum number of spikes for all neurons
% in simulation. Make it 50Hz across all neurons
% If there are more spikes, the simulation will
% terminate
maxns=ceil(.05*N*Tburn);
jistim=0;
stimProb = 0;
StimulatedNeurons = binornd(1,stimProb, Ne,1);
jestim=0;


% Stimulate a subpopulation of E neurons only.
Jstim=sqrt(N)*[jestim*StimulatedNeurons ; jistim*ones(Ni,1)]; % Stimulate only E neurons
Istim=zeros(size(timeBP));


%%% Make (correlated) Poisson spike times for ffwd layer
%%% See section 5 of SimDescription.pdf for a description of this algorithm
tic
if(c<1e-5) % If uncorrelated
    nspikeX=poissrnd(Nx*rx*Tburn);
    st=rand(nspikeX,1)*Tburn;
    sx=zeros(2,numel(st));
    sx(1,:)=sort(st);
    sx(2,:)=randi(Nx,1,numel(st)); % neuron indices
    clear st;
else % If correlated
    rm=rx/c; % Firing rate of mother process
    nstm=poissrnd(rm*Tburn); % Number of mother spikes
    stm=rand(nstm,1)*Tburn; % spike times of mother process
    maxnsx=Tburn*rx*Nx*1.2; % Max num spikes
    sx=zeros(2,maxnsx);
    ns=0;
    for j=1:Nx  % For each ffwd spike train
        ns0=binornd(nstm,c); % Number of spikes for this spike train
        st=randsample(stm,ns0); % Sample spike times randomly
        st=st+taujitter*randn(size(st)); % jitter spike times
        st=st(st>0 & st<Tburn); % Get rid of out-of-bounds times
        ns0=numel(st); % Re-compute spike count
        sx(1,ns+1:ns+ns0)=st; % Set the spike times and indices
        sx(2,ns+1:ns+ns0)=j;
        ns=ns+ns0;
    end
    
    % Get rid of padded zeros
    sx = sx(:,sx(1,:)>0);
    
    % Sort by spike time
    [~,I] = sort(sx(1,:));
    sx = sx(:,I);
    
    
    nspikeX=size(sx,2);
    
end
tGenx=toc;
disp(sprintf('\nTime to generate ffwd spikes for burn-in period: %.2f sec',tGenx))


% Random initial voltages
V0=rand(N,1)*(VT-Vre)+Vre;

V=V0;
Ie=zeros(N,1);
Ii=zeros(N,1);
Ix=zeros(N,1);
x=zeros(N,1);
%IeRec=zeros(numrecord,Ntrec);
%IiRec=zeros(numrecord,Ntrec);
%IxRec=zeros(numrecord,Ntrec);
%VRec=zeros(numrecord,Ntrec);
%wRec=zeros(numrecord,Ntrec);
JRec=zeros(numrecordJ_ei,Ntrec);
iFspike=1;
nspike=0;
TooManySpikes=0;
tic
BurnS=zeros(2,maxns);

for i=1:numel(timeBP)
    
    
    % Propogate ffwd spikes
    while(sx(1,iFspike)<=timeBP(i) && iFspike<nspikeX)
        jpre=sx(2,iFspike);
        Ix=Ix+Jx(:,jpre)/taux;
        iFspike=iFspike+1;
    end
    
    
    % Euler update to V
    V=V+(dt/Cm)*(Istim(i)*Jstim+Ie+Ii+Ix+gL*(EL-V)+gL*DeltaT*exp((V-VT)/DeltaT));
    
    % Find which neurons spiked
    Ispike=find(V>=Vth);
    
    % If there are spikes
    if(~isempty(Ispike))
        
        % Store spike times and neuron indices
        if(nspike+numel(Ispike)<=maxns)
            BurnS(1,nspike+1:nspike+numel(Ispike))=timeBP(i);
            BurnS(2,nspike+1:nspike+numel(Ispike))=Ispike;
        else
            TooManySpikes=1;
            break;
        end
        
        
        % Update synaptic currents
        Ie=Ie+sum(J(:,Ispike(Ispike<=Ne)),2)/taue;
        Ii=Ii+sum(J(:,Ispike(Ispike>Ne)),2)/taui;
        
        % If there is EE Hebbian plasticity
        if(eta_ee~=0)
            %Update synaptic weights according to plasticity rules
            %E to E after presynaptic spike
            J(1:Ne,Ispike(Ispike<=Ne))=J(1:Ne,Ispike(Ispike<=Ne))+ ...
                -repmat(eta_ee*(x(1:Ne)),1,nnz(Ispike<=Ne)).*(J(1:Ne,Ispike(Ispike<=Ne)));
            %E to E after a postsynaptic spike
            J(Ispike(Ispike<=Ne),1:Ne)=J(Ispike(Ispike<=Ne),1:Ne)+ ...
                +repmat(eta_ee*x(1:Ne)',nnz(Ispike<=Ne),1).*(Jmax_ee).*(J(Ispike(Ispike<=Ne),1:Ne)~=0);
        end
        
        % If there is EI plasticity
        if(eta_ei~=0)
            % Update synaptic weights according to plasticity rules
            % I to E after an I spike
            J(1:Ne,Ispike(Ispike>Ne))=J(1:Ne,Ispike(Ispike>Ne))+ ...
                -repmat(eta_ei*(x(1:Ne)-alpha_e),1,nnz(Ispike>Ne)).*(J(1:Ne,Ispike(Ispike>Ne)));
            % E to I after an E spike
            J(Ispike(Ispike<=Ne),Ne+1:N)=J(Ispike(Ispike<=Ne),Ne+1:N)+ ...
                -repmat(eta_ei*x(Ne+1:N)',nnz(Ispike<=Ne),1).*(J(Ispike(Ispike<=Ne),Ne+1:N));
        end
        
        % If there is IE *Homeostatic* plasticity
        if(eta_ie_homeo~=0)
            % Update synaptic weights according to plasticity rules
            % after an E spike    
            J(Ne+1:N,Ispike(Ispike<=Ne))=J(Ne+1:N,Ispike(Ispike<=Ne))+ ... 
                -repmat(eta_ie_homeo*(x(Ne+1:N)-alpha_ie),1,nnz(Ispike<=Ne)).*(J(Ne+1:N,Ispike(Ispike<=Ne)));
            % I to E after an I spike
            J(Ispike(Ispike>Ne),1:Ne)=J(Ispike(Ispike>Ne),1:Ne)+ ... 
                -repmat(eta_ie_homeo*x(1:Ne)',nnz(Ispike>Ne),1).*(J(Ispike(Ispike>Ne),1:Ne));
        end
        
        % If there is IE *Hebbian* plasticity
        if(eta_ie_hebbian~=0)
            %Update synaptic weights according to plasticity rules
            %E to E after presynaptic spike
            J(Ne+1:N,Ispike(Ispike<=Ne))=J(Ne+1:N,Ispike(Ispike<=Ne))+ ...
                -repmat(eta_ie_hebbian*(x(Ne+1:N)),1,nnz(Ispike<=Ne)).*(J(Ne+1:N,Ispike(Ispike<=Ne)));
            %E to E after a postsynaptic spike
            J(Ispike(Ispike>Ne),1:Ne)=J(Ispike(Ispike>Ne),1:Ne)+ ...
                +repmat(eta_ie_hebbian*x(1:Ne)',nnz(Ispike>Ne),1).*(Jmax_ie).*(J(Ispike(Ispike>Ne),1:Ne)~=0);
        end
        
        % If there is II Homeostatic plasticity
        if(eta_ii~=0)
            % Update synaptic weights according to plasticity rules
            % I to I after a presyanptic spike    
            J(Ne+1:N,Ispike(Ispike>Ne))=J(Ne+1:N,Ispike(Ispike>Ne))+ ... 
                -repmat(eta_ii*(x(Ne+1:N)-alpha_ii),1,nnz(Ispike>Ne)).*(J(Ne+1:N,Ispike(Ispike>Ne)));
            % I to I after a postsynaptic spike
            J(Ispike(Ispike>Ne),Ne+1:N)=J(Ispike(Ispike>Ne),Ne+1:N)+ ... 
                -repmat(eta_ii*x(Ne+1:N)',nnz(Ispike>Ne),1).*(J(Ispike(Ispike>Ne),Ne+1:N));
        end
        
        % Update rate estimates for plasticity rules
        x(Ispike)=x(Ispike)+1;
        
        % Update cumulative number of spikes
        nspike=nspike+numel(Ispike);
    end
    
    % Euler update to synaptic currents
    Ie=Ie-dt*Ie/taue;
    Ii=Ii-dt*Ii/taui;
    Ix=Ix-dt*Ix/taux;
    
    % Update time-dependent firing rates for plasticity
    x(1:Ne)=x(1:Ne)-dt*x(1:Ne)/tauSTDP;
    x(Ne+1:end)=x(Ne+1:end)-dt*x(Ne+1:end)/tauSTDP;
    
    % This makes plots of V(t) look better.
    % All action potentials reach Vth exactly.
    % This has no real effect on the network sims
    V(Ispike)=Vth;
    
    % Store recorded variables
    ii=IntDivide(i,nBinsRecord);
    %IeRec(:,ii)=IeRec(:,ii)+Ie(Irecord);
    %IiRec(:,ii)=IiRec(:,ii)+Ii(Irecord);
    %IxRec(:,ii)=IxRec(:,ii)+Ix(Irecord);
    %VRec(:,ii)=VRec(:,ii)+V(Irecord);
    JRec(:,ii)=J(sub2ind(size(J),Jrecord_ei(1,:),Jrecord_ei(2,:)));
    
    % Reset mem pot.
    V(Ispike)=Vre;
    
    
end
%IeRec=IeRec/nBinsRecord; % Normalize recorded variables by # bins
%IiRec=IiRec/nBinsRecord;
%IxRec=IxRec/nBinsRecord;
%VRec=VRec/nBinsRecord;
BurnS=BurnS(:,1:nspike); % Get rid of padding in s
tSim=toc;
disp(sprintf('\nTime for simulation: %.2f min',tSim/60))


% Mean rate of each neuron (excluding burn-in period)
%Tburnburn=200;
%reSim=hist( BurnS(2,BurnS(1,:)>Tburnburn & BurnS(2,:)<=Ne),1:Ne)/(Tburn-Tburnburn);
%riSim=hist( BurnS(2,BurnS(1,:)>Tburnburn & BurnS(2,:)>Ne)-Ne,1:Ni)/(Tburn-Tburnburn);


% Time-dependent mean rates
dtRate=100;
eRateT=hist(BurnS(1,BurnS(2,:)<=Ne),1:dtRate:Tburn)/(dtRate*Ne);
iRateT=hist(BurnS(1,BurnS(2,:)>Ne),1:dtRate:Tburn)/(dtRate*Ni);

% % Plot time-dependent rates
% figure;
% plot((dtRate:dtRate:Tburn)/1000,1000*eRateT)
% hold on
% plot((dtRate:dtRate:Tburn)/1000,1000*iRateT)
% legend('r_e','r_i')
% ylabel('rate (Hz)')
% xlabel('time (s)')
% title('Network"s burn in period');
% 
% % If plastic, plot mean connection weights over time
% if(eta~=0)
%    figure
%    plot(timeRecord/1000,mean(JRec)*sqrt(N))
%    xlabel('time (s)')
%    ylabel('Mean I to E synaptic weight')
%    title('Weight evolution in burn-in period')
% end

%% Burn in period code ends. Now that the network is at steady state, the
% trials can start.


%%% All the code below computes spike count covariances and correlations
%%% for BURN IN period.
%%% We want to compare the resulting covariances to what is predicted by
%%% the theory.

% Compute spike count covariances over windows of size
% winsize starting at time T1 and ending at time T2.
winsize=200;
T1=Tburn/2; % Burn-in period of 250 ms
T2=Tburn;   % Compute covariances until end of simulation
C=SpikeCountCov(BurnS,N,T1,T2,winsize);
R = corrcov(C);

% Get mean spike count covariances over each sub-pop
[II,JJ]=meshgrid(1:N,1:N);
mCeeBurn=mean(C(II<=Ne & JJ<=II));
mCeiBurn=mean(C(II<=Ne & JJ>Ne));
mCiiBurn=mean(C(II>Ne & JJ>II));
mC_BurnPeriod = mCeeBurn + 2* mCeiBurn + mCiiBurn;
mReeBurn=mean(R(II<=Ne & JJ<=II & isfinite(R)));
mReiBurn=mean(R(II<=Ne & JJ>Ne & isfinite(R)));
mRiiBurn=mean(R(II>Ne & JJ>II & isfinite(R)));
mR_BurnPeriod = [mReeBurn mReiBurn ; mReiBurn mRiiBurn];
sprintf('Mean correlations of burn-in period is: %f', mR_BurnPeriod);
clear Tburn

%% %%%%%%%%%%% Start of the experiment.%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trial Configuration.
Total_number_trials = 120; % Number of trials laser and control. 
% Half of this is control, half laser.

ControlTrials = sort(randsample(Total_number_trials, Total_number_trials/2, false)) ; % Let the order of
% control and stim trials be random in a block. If we have this
% outside of the loop, then all blocks have the same sequence. To change
% this, put ControlTrial inside the BlockTrial loop or make it a matrix.
LaserTrials = setdiff([1:Total_number_trials],ControlTrials);

% Time (in ms) for sim
TrialTime = 1400; % We have 1s to let the network reach steady state.
% then we have 300ms of stimulation, and the trial ends 1.7 sec later.
StimDuration = 300; % ms.
T=TrialTime*Total_number_trials;

% Number of time bins to average over when recording
nBinsRecord=10;
dtRecord=nBinsRecord*dt;
timeRecord=dtRecord:dtRecord:T;
Ntrec=numel(timeRecord);

% Number of time bins
Nt=round(T/dt);
time=dt:dt:T;

% Preallocate memory.
reMean = zeros(1,Total_number_trials);
riMean = zeros(1,Total_number_trials);
reLaserPeriod = zeros(1,Total_number_trials);
riLaserPeriod = zeros(1,Total_number_trials);
rePostLaserPeriod = zeros(1,Total_number_trials);
riPostLaserPeriod = zeros(1,Total_number_trials);
std_E_LP = zeros(1,Total_number_trials);
std_I_LP = zeros(1,Total_number_trials);
std_E_NLP = zeros(1,Total_number_trials);
std_I_NLP = zeros(1,Total_number_trials);

% Maximum number of spikes for all neurons
% in simulation. Make it 50Hz across all neurons
% If there are more spikes, the simulation will
% terminate
maxns=ceil(.05*N*T);
jistim=0;
stimProb = 0.50;
StimulatedNeurons = [ones(Ne*stimProb,1); zeros(Ne*(1-stimProb),1)];  % binornd(1,stimProb, Ne,1);
jestim=0.1;
laserOnSet = 500;

% Set the laser frequency.
laser_freq = 100; % Hz

if laser_freq == 35
    pulseWidth = 10; % Pulse width.
    timebetweenPulse = 19; %Time between pulses.
    Cycles = 10;
elseif laser_freq == 10
    pulseWidth = 33; % Pulse width.
    timebetweenPulse = 66; %Time between pulses.
    Cycles = 3;
elseif laser_freq == 20
    pulseWidth = 16; % Pulse width.
    timebetweenPulse = 33; %Time between pulses.
    Cycles = 6;
elseif laser_freq == 100
    pulseWidth = 3.3; % Pulse width.
    timebetweenPulse = 6.6; %Time between pulses.
    Cycles = 30;
end


% % Synaptic weights to record.
% % The first row of Jrecord is the postsynaptic indices
% % The second row is the presynaptic indices
% nJrecord0=1000; % Number to record
% [II,JJ]=find(J(find(StimulatedNeurons==1)',Ne+1:N)); % Find non-zero I to E weights of stim neurons
% III=randperm(numel(II),nJrecord0); % Choose some at random to record
% II=II(III);
% JJ=JJ(III);
% Jrecord=[II JJ+Ne]'; % Record these
% numrecordJ=size(Jrecord,2);
% if(size(Jrecord,1)~=2)
%     error('Jrecord must be 2xnumrecordJ');
% end

% Synaptic weights EE to record.
% The first row of Jrecord is the postsynaptic indices
% The second row is the presynaptic indices
nJrecord1=2000; % Number to record
[II1,JJ1]=find(J(1:Ne,1:Ne)); % Find non-zero E to E weights of stim neurons
III1=randperm(numel(II1),nJrecord1); % Choose some at random to record
II1=II1(III1);
JJ1=JJ1(III1);
Jrecord_ee=[II1 JJ1]'; % Record these
numrecordJ_ee=size(Jrecord_ee,2);
if(size(Jrecord_ee,1)~=2)
    error('Jrecord must be 2xnumrecordJ');
end

% Synaptic weights EI to record.
% The first row of Jrecord is the postsynaptic indices
% The second row is the presynaptic indices
nJrecord0=2000; % Number to record
[II,JJ]=find(J(1:Ne,Ne+1:N)); % Find non-zero I to E weights of stim neurons
III=randperm(numel(II),nJrecord0); % Choose some at random to record
II=II(III);
JJ=JJ(III);
Jrecord_ei=[II JJ+Ne]'; % Record these
numrecordJ_ei=size(Jrecord_ei,2);
if(size(Jrecord_ei,1)~=2)
    error('Jrecord must be 2xnumrecordJ');
end

% Synaptic weights IE to record.
% The first row of Jrecord is the postsynaptic indices
% The second row is the presynaptic indices
nJrecord0=2000; % Number to record
[II,JJ]=find(J(Ne+1:N,1:Ne)); % Find non-zero E to I weights of stim neurons
III=randperm(numel(II),nJrecord0); % Choose some at random to record
II=II(III);
JJ=JJ(III);
Jrecord_ie=[II+Ne JJ]'; % Record these
numrecordJ_ie=size(Jrecord_ie,2);
if(size(Jrecord_ie,1)~=2)
    error('Jrecord must be 2xnumrecordJ');
end

% Synaptic weights II to record.
% The first row of Jrecord is the postsynaptic indices
% The second row is the presynaptic indices
nJrecord0=2000; % Number to record
[II,JJ]=find(J(Ne+1:N,Ne+1:N)); % Find non-zero I to I weights of stim neurons
III=randperm(numel(II),nJrecord0); % Choose some at random to record
II=II(III);
JJ=JJ(III);
Jrecord_ii=[II+Ne JJ+Ne]'; % Record these
numrecordJ_ii=size(Jrecord_ii,2);
if(size(Jrecord_ii,1)~=2)
    error('Jrecord must be 2xnumrecordJ');
end


% Now that the experiment starts, we set a higher threshold.
Jmax_ee = 15/sqrt(N);
Jmax_ie = 150/sqrt(N); % only if hebbian


s=zeros(2,maxns);

% Extra stimulus: Istim is a time-dependent stimulus
% it is delivered to all neurons with weights given by JIstim.
% Specifically, the stimulus to neuron j at time index i is:
% Istim(i)*JIstim(j)

Istim=zeros(size(time));

for i = 1:Total_number_trials
    
    if(ismember(i,ControlTrials))
        % Do nothing. So that the Istim in this control trial is zero.
    else
        %Istim(time>1000+(i-1)*TrialTime & time<1000+StimDuration+(i-1)*TrialTime)=1; %This is for step
        %input.
        % Next three lines are for pulsed input.
        for j = 1:Cycles
            Istim( time>laserOnSet+(i-1)*TrialTime+(j-1)*(pulseWidth+timebetweenPulse) ...
                & time<laserOnSet+(j-1)*(pulseWidth+timebetweenPulse)+(i-1)*TrialTime+pulseWidth) = 1;
        end
    end
end

% Stimulate a subpopulation of E neurons only.
Jstim=sqrt(N)*[jestim*StimulatedNeurons ; jistim*ones(Ni,1)]; % Stimulate only E neurons



%%% Make (correlated) Poisson spike times for ffwd layer
%%% See section 5 of SimDescription.pdf for a description of this algorithm
tic
if(c<1e-5) % If uncorrelated
    nspikeX=poissrnd(Nx*rx*T);
    st=rand(nspikeX,1)*T;
    sx=zeros(2,numel(st));
    sx(1,:)=sort(st);
    sx(2,:)=randi(Nx,1,numel(st)); % neuron indices
    clear st;
else % If correlated
    rm=rx/c; % Firing rate of mother process
    nstm=poissrnd(rm*T); % Number of mother spikes
    stm=rand(nstm,1)*T; % spike times of mother process
    maxnsx=T*rx*Nx*1.2; % Max num spikes
    sx=zeros(2,maxnsx);
    ns=0;
    for j=1:Nx  % For each ffwd spike train
        ns0=binornd(nstm,c); % Number of spikes for this spike train
        st=randsample(stm,ns0); % Sample spike times randomly
        st=st+taujitter*randn(size(st)); % jitter spike times
        st=st(st>0 & st<T); % Get rid of out-of-bounds times
        ns0=numel(st); % Re-compute spike count
        sx(1,ns+1:ns+ns0)=st; % Set the spike times and indices
        sx(2,ns+1:ns+ns0)=j;
        ns=ns+ns0;
    end
    
    % Get rid of padded zeros
    sx = sx(:,sx(1,:)>0);
    
    % Sort by spike time
    [~,I] = sort(sx(1,:));
    sx = sx(:,I);
    
    
    nspikeX=size(sx,2);
    
end
tGenx=toc;
disp(sprintf('\nTime to generate ffwd spikes: %.2f sec',tGenx))


% Random initial voltages
V0=rand(N,1)*(VT-Vre)+Vre;

V=V0;
Ie=zeros(N,1);
Ii=zeros(N,1);
Ii_stim=zeros(Ne/2,1);
Ix=zeros(N,1);
x=zeros(N,1);
IeRec=zeros(1,Ntrec);
IiRec=zeros(1,Ntrec);
IiRec_stim=zeros(1,Ntrec);
IxRec=zeros(1,Ntrec);
%     VRec=zeros(numrecord,Ntrec);
%     wRec=zeros(numrecord,Ntrec);
JRec_ee=zeros(1,Ntrec);
JRec_ei=zeros(1,Ntrec);
JRec_ie=zeros(1,Ntrec);
JRec_ii=zeros(1,Ntrec);
JRec_ee_std=zeros(1,Ntrec);
JRec_ei_std=zeros(1,Ntrec);
JRec_ie_std=zeros(1,Ntrec);
JRec_ii_std=zeros(1,Ntrec);
iFspike=1;
nspike=0;
TooManySpikes=0;
tic
for i=1:numel(time)
    
    
    % Propogate ffwd spikes
    while(sx(1,iFspike)<=time(i) && iFspike<nspikeX)
        jpre=sx(2,iFspike);
        Ix=Ix+Jx(:,jpre)/taux;
        iFspike=iFspike+1;
    end
    
    
    % Euler update to V
    V=V+(dt/Cm)*(Istim(i)*Jstim+Ie+Ii+Ix+gL*(EL-V)+gL*DeltaT*exp((V-VT)/DeltaT));
    
    % Find which neurons spiked
    Ispike=find(V>=Vth);
    
    % If there are spikes
    if(~isempty(Ispike))
        
        % Store spike times and neuron indices
        if(nspike+numel(Ispike)<=maxns)
            s(1,nspike+1:nspike+numel(Ispike))=time(i);
            s(2,nspike+1:nspike+numel(Ispike))=Ispike;
            
        else
            TooManySpikes=1;
            break;
        end
        
        
        % Update synaptic currents
        Ie=Ie+sum(J(:,Ispike(Ispike<=Ne)),2)/taue;
        Ii=Ii+sum(J(:,Ispike(Ispike>Ne)),2)/taui;
        Ii_stim=Ii_stim+sum(J(1:Ne/2,Ispike(Ispike>Ne)),2)/taui;

        % If there is EE Hebbian plasticity
        if(eta_ee~=0)
            %Update synaptic weights according to plasticity rules
            %E to E after presynaptic spike
            J(1:Ne,Ispike(Ispike<=Ne))=J(1:Ne,Ispike(Ispike<=Ne))+ ...
                -repmat(eta_ee*(x(1:Ne)),1,nnz(Ispike<=Ne)).*(J(1:Ne,Ispike(Ispike<=Ne)));
            %E to E after a postsynaptic spike
            J(Ispike(Ispike<=Ne),1:Ne)=J(Ispike(Ispike<=Ne),1:Ne)+ ...
                +repmat(eta_ee*x(1:Ne)',nnz(Ispike<=Ne),1).*(Jmax_ee).*(J(Ispike(Ispike<=Ne),1:Ne)~=0);
        end
        
        % If there is EI plasticity
        if(eta_ei~=0)
            % Update synaptic weights according to plasticity rules
            % I to E after an I spike
            J(1:Ne,Ispike(Ispike>Ne))=J(1:Ne,Ispike(Ispike>Ne))+ ...
                -repmat(eta_ei*(x(1:Ne)-alpha_e),1,nnz(Ispike>Ne)).*(J(1:Ne,Ispike(Ispike>Ne)));
            % E to I after an E spike
            J(Ispike(Ispike<=Ne),Ne+1:N)=J(Ispike(Ispike<=Ne),Ne+1:N)+ ...
                -repmat(eta_ei*x(Ne+1:N)',nnz(Ispike<=Ne),1).*(J(Ispike(Ispike<=Ne),Ne+1:N));
        end
        
        % If there is IE *Homeostatic* plasticity
        if(eta_ie_homeo~=0)
            % Update synaptic weights according to plasticity rules
            % after an E spike    
            J(Ne+1:N,Ispike(Ispike<=Ne))=J(Ne+1:N,Ispike(Ispike<=Ne))+ ... 
                -repmat(eta_ie_homeo*(x(Ne+1:N)-alpha_ie),1,nnz(Ispike<=Ne)).*(J(Ne+1:N,Ispike(Ispike<=Ne)));
            % I to E after an I spike
            J(Ispike(Ispike>Ne),1:Ne)=J(Ispike(Ispike>Ne),1:Ne)+ ... 
                -repmat(eta_ie_homeo*x(1:Ne)',nnz(Ispike>Ne),1).*(J(Ispike(Ispike>Ne),1:Ne));
        end
        
        % If there is IE *Hebbian* plasticity
        if(eta_ie_hebbian~=0)
            %Update synaptic weights according to plasticity rules
            %E to E after presynaptic spike
            J(Ne+1:N,Ispike(Ispike<=Ne))=J(Ne+1:N,Ispike(Ispike<=Ne))+ ...
                -repmat(eta_ie_hebbian*(x(Ne+1:N)),1,nnz(Ispike<=Ne)).*(J(Ne+1:N,Ispike(Ispike<=Ne)));
            %E to E after a postsynaptic spike
            J(Ispike(Ispike>Ne),1:Ne)=J(Ispike(Ispike>Ne),1:Ne)+ ...
                +repmat(eta_ie_hebbian*x(1:Ne)',nnz(Ispike>Ne),1).*(Jmax_ie).*(J(Ispike(Ispike>Ne),1:Ne)~=0);
        end
        
        % If there is II Homeostatic plasticity
        if(eta_ii~=0)
            % Update synaptic weights according to plasticity rules
            % I to I after a presyanptic spike    
            J(Ne+1:N,Ispike(Ispike>Ne))=J(Ne+1:N,Ispike(Ispike>Ne))+ ... 
                -repmat(eta_ii*(x(Ne+1:N)-alpha_ii),1,nnz(Ispike>Ne)).*(J(Ne+1:N,Ispike(Ispike>Ne)));
            % I to I after a postsynaptic spike
            J(Ispike(Ispike>Ne),Ne+1:N)=J(Ispike(Ispike>Ne),Ne+1:N)+ ... 
                -repmat(eta_ii*x(Ne+1:N)',nnz(Ispike>Ne),1).*(J(Ispike(Ispike>Ne),Ne+1:N));
        end
        
        
        % Update rate estimates for plasticity rules
        x(Ispike)=x(Ispike)+1;
        
        % Update cumulative number of spikes
        nspike=nspike+numel(Ispike);
    end
    
    % Euler update to synaptic currents
    Ie=Ie-dt*Ie/taue;
    Ii=Ii-dt*Ii/taui;
    Ii_stim=Ii_stim-dt*Ii_stim/taui;
    Ix=Ix-dt*Ix/taux;
    
    % Update time-dependent firing rates for plasticity
    x(1:Ne)=x(1:Ne)-dt*x(1:Ne)/tauSTDP;
    x(Ne+1:end)=x(Ne+1:end)-dt*x(Ne+1:end)/tauSTDP;
    
    % This makes plots of V(t) look better.
    % All action potentials reach Vth exactly.
    % This has no real effect on the network sims
    V(Ispike)=Vth;
    
    % Store recorded variables
    ii=IntDivide(i,nBinsRecord);
    IeRec(1,ii)=IeRec(1,ii)+mean(Ie(Irecord));
    IiRec(1,ii)=IiRec(1,ii)+mean(Ii(Irecord));
    IiRec_stim(1,ii)=IiRec_stim(1,ii)+mean(Ii_stim);
    IxRec(1,ii)=IxRec(1,ii)+mean(Ix(Irecord));
    %VRec(:,ii)=VRec(:,ii)+V(Irecord);
    JRec_ee(1,ii)=mean(J(sub2ind(size(J),Jrecord_ee(1,:),Jrecord_ee(2,:))));
    JRec_ei(1,ii)=mean(J(sub2ind(size(J),Jrecord_ei(1,:),Jrecord_ei(2,:))));
    JRec_ie(1,ii)=mean(J(sub2ind(size(J),Jrecord_ie(1,:),Jrecord_ie(2,:))));
    JRec_ii(1,ii)=mean(J(sub2ind(size(J),Jrecord_ii(1,:),Jrecord_ii(2,:))));
    JRec_ee_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ee(1,:),Jrecord_ee(2,:))));
    JRec_ei_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ei(1,:),Jrecord_ei(2,:))));
    JRec_ie_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ie(1,:),Jrecord_ie(2,:))));
    JRec_ii_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ii(1,:),Jrecord_ii(2,:))));

    % Reset mem pot.
    V(Ispike)=Vre;
    
end
IeCurrent=IeRec/nBinsRecord; % Normalize recorded variables by # bins
IiCurrent=IiRec/nBinsRecord;
IiCurrent_stim=IiRec_stim/nBinsRecord;
IxCurrent=IxRec/nBinsRecord;
%VRec=VRec/nBinsRecord;
% JRec=JRec/nBinsRecord;

s=s(:,1:nspike); % Get rid of padding in s
tSim=toc;
disp(sprintf('\nTime for simulation: %.2f min',tSim/60))


% Mean rate of each neuron (excluding burn-in period)
Tburn=0;
%reSim=hist( s{BlockTrial}(2,s{BlockTrial}(1,:)>Tburn & s{BlockTrial}(2,:)<=Ne),1:Ne)/(T-Tburn);
%riSim=hist( s{BlockTrial}(2,s{BlockTrial}(1,:)>Tburn & s{BlockTrial}(2,:)>Ne)-Ne,1:Ni)/(T-Tburn);

% Next, calculate the rate of E and I for each single trial.
for i = 1:Total_number_trials
    reSim=hist( s( 2,s(1,:)>TrialTime*(i-1) & s(2,:)<=Ne ...
        & s(1,:)< TrialTime*i ),1:Ne)/(TrialTime);
    riSim=hist( s( 2,s(1,:)>Tburn+TrialTime*(i-1) & s(2,:)>Ne ...
        & s(1,:)< TrialTime*i)-Ne,1:Ni)/(TrialTime-Tburn);
    reMean(1, i)=mean(reSim); % Mean rate of E neurons at some Trial 'i' in a Block.
    riMean(1, i)=mean(riSim); % Mean rate of I neurons at some Trial 'i' in a Block.
    
    % Calculate the rates at each trial and block of laser and
    % nonlaser pairs separately.
    rate_LaserPairs(1, i) = mean( hist( s( 2,s(1,:)>Tburn+TrialTime*(i-1) & ismember(s(2,:),find(StimulatedNeurons==1)) ...
        & s(1,:)< TrialTime*i ),1:sum(StimulatedNeurons))/(TrialTime-Tburn) );
    rate_nonLaserPairs(1, i) = mean( hist( s( 2,s(1,:)>Tburn+TrialTime*(i-1) & ismember(s(2,:),find(StimulatedNeurons==0)) ...
        & s(1,:)< TrialTime*i ),1:(N-sum(StimulatedNeurons)))/(TrialTime-Tburn) );
    
    % Calulcate PSTH over laser pairs.
    dtRate=50;
    spikeTimes = s(1, ismember(s(2,:),find(StimulatedNeurons==1)));
    
    eRateTime_TRIAL{1,i}=hist(spikeTimes(spikeTimes>TrialTime*(i-1) & spikeTimes< TrialTime*i)...
        ,TrialTime*(i-1)+1:dtRate:TrialTime*i)/(dtRate*sum(StimulatedNeurons));
    
    iRateTime_TRIAL{1,i}=hist(s(1,s(1,:)>TrialTime*(i-1) & ismember(s(2,:),find(StimulatedNeurons==0))...
        & s(1,:)< TrialTime*i ),TrialTime*(i-1)+1:dtRate:TrialTime*i)/(dtRate*(N-sum(StimulatedNeurons)));
    
    
    % Calculate the mean rates over the LASER PERIOD and POST LASER PERIOD.
    reSim=hist( s( 2,s(1,:)>Tburn+TrialTime*(i-1)+laserOnSet & s(2,:)<=Ne...
        & s(1,:)< TrialTime*i+(laserOnSet+300) ),1:Ne)/(TrialTime-Tburn);
    riSim=hist( s( 2,s(1,:)>Tburn+TrialTime*(i-1)+laserOnSet & s(2,:)>Ne...
        & s(1,:)< TrialTime*i+(laserOnSet+300) )-Ne,1:Ni)/(TrialTime-Tburn); % 300 is duration of the laser
    reLaserPeriod(1,i) = mean(reSim);
    riLaserPeriod(1,i) = mean(riSim);
    std_E_LP(1,i) = std(reSim)/sqrt(Ne);
    std_I_LP(1,i) = std(riSim)/sqrt(Ni);
    
    
    % POST LASER PERIOD.
    reSim=hist( s( 2,s(1,:)>Tburn+TrialTime*(i-1)+(laserOnSet+300) & s(2,:)<=Ne & s(1,:)< TrialTime*i+(laserOnSet+600) ),1:Ne)/(TrialTime-Tburn);
    riSim=hist( s( 2,s(1,:)>Tburn+TrialTime*(i-1)+(laserOnSet+300) & s(2,:)>Ne & s(1,:)< TrialTime*i+(laserOnSet+600) )-Ne,1:Ni)/(TrialTime-Tburn);
    rePostLaserPeriod(1,i) = mean(reSim);
    riPostLaserPeriod(1,i) = mean(riSim);
    std_E_NLP(1,i) = std(reSim)/sqrt(Ne);
    std_I_NLP(1,i) = std(riSim)/sqrt(Ni);
    
end

% iii=1;
% jjj=1;
% FR_laserPairs_Control = zeros(1,TrialTime/dtRate);
% FR_laserPairs_Laser = zeros(1,TrialTime/dtRate);
% for i = 1:Total_number_trials
%     if(ismember(i,ControlTrials)) % Count spikes only if i is a control trial.
%         FR_laserPairs_Control = FR_laserPairs_Control + eRateTime_TRIAL{1,i};
%         iii = iii+1;
%     else
%         FR_laserPairs_Laser = FR_laserPairs_Laser + eRateTime_TRIAL{1,i};
%         jjj=jjj+1;
%     end
% end
%
% FR_laserPairs_Control = FR_laserPairs_Control / iii;
% FR_laserPairs_Laser = FR_laserPairs_Laser / jjj;

%% Compute the rates sliding a window over time and trials.
winsize=200; %ms.
sliding_window = 50; %ms.
Number_of_points_rates = (TrialTime/winsize-1)*(winsize/sliding_window); % Number of
% points in the time axis for which correlations are calculated.

block_slide_size = 20; % = how many trials in one block.
NBlockTrials = length(LaserTrials)-block_slide_size;


FR_laserPairs_Control = zeros(NBlockTrials,Number_of_points_rates);
FR_laserPairs_Laser = zeros(NBlockTrials,Number_of_points_rates);

spikeTimes = s(1, ismember(s(2,:),find(StimulatedNeurons==1)));
for BlockTrial=1:NBlockTrials
    tic
    ControlTrialsUsed = ControlTrials(BlockTrial:block_slide_size+BlockTrial-1);
    LaserTrialsUsed = LaserTrials(BlockTrial:block_slide_size+BlockTrial-1);
    for i = 1:Total_number_trials
        if(ismember(i,ControlTrialsUsed)) % Count spikes only if i is a control trial.
            for window = 1:Number_of_points_rates
                T1=TrialTime*(i-1) + sliding_window*(window-1); % Burn-in period of 200 ms
                T2=T1+winsize;   % Compute covariances until end of simulation
                
                FR_laserPairs_Control(BlockTrial,window) = FR_laserPairs_Control(BlockTrial,window) +...
                    histcounts(spikeTimes(spikeTimes>T1 & spikeTimes< T2)...
                    ,T1:winsize:T2)/(winsize*sum(StimulatedNeurons));
            end
        elseif(ismember(i,LaserTrialsUsed))  % if laser trial.
            for window = 1:Number_of_points_rates
                T1=TrialTime*(i-1) + sliding_window*(window-1); % Burn-in period of 200 ms
                T2=T1+winsize;   % Compute covariances until end of simulation
                
                FR_laserPairs_Laser(BlockTrial,window) = FR_laserPairs_Laser(BlockTrial,window) +...
                    histcounts(spikeTimes(spikeTimes>T1 & spikeTimes< T2)...
                    ,T1:winsize:T2)/(winsize*sum(StimulatedNeurons));
            end
        end
        
    end
    toc
end
% We summed spike counts over trials, now we divide by over how many trials
% we added them.
FR_laserPairs_Control = FR_laserPairs_Control / block_slide_size; 
FR_laserPairs_Laser = FR_laserPairs_Laser / block_slide_size;


%% Compute the PSTHs sliding a GAUSSIAN window over time and trials.
winsize1=1; %ms.
sliding_window1 = 1; %ms.
Number_of_points_PSTH = (TrialTime/winsize1-1)*(winsize1/sliding_window1); % Number of
% points in the time axis for which correlations are calculated.

block_slide_size = 20; % = how many trials in one block.
NBlockTrials = length(LaserTrials)-block_slide_size;

PSTH_laserPairs_Control = zeros(NBlockTrials,Number_of_points_PSTH);
PSTH_laserPairs_Laser = zeros(NBlockTrials,Number_of_points_PSTH);

spikeTimes = s(1, ismember(s(2,:),find(StimulatedNeurons==1)));
for BlockTrial=1:NBlockTrials
    tic
    ControlTrialsUsed = ControlTrials(BlockTrial:block_slide_size+BlockTrial-1);
    LaserTrialsUsed = LaserTrials(BlockTrial:block_slide_size+BlockTrial-1);
    for i = 1:Total_number_trials
        if(ismember(i,ControlTrialsUsed)) % Count spikes only if i is a control trial.
            for window = 1:Number_of_points_PSTH
                T1=TrialTime*(i-1) + sliding_window1*(window-1); % Burn-in period of 200 ms
                T2=T1+winsize1;   % Compute covariances until end of simulation
                
                PSTH_laserPairs_Control(BlockTrial,window) = PSTH_laserPairs_Control(BlockTrial,window) +...
                    histcounts(spikeTimes(spikeTimes>T1 & spikeTimes< T2)...
                    ,T1:winsize1:T2)/(winsize1*sum(StimulatedNeurons));
            end
        elseif(ismember(i,LaserTrialsUsed))  % if laser trial.
            for window = 1:Number_of_points_PSTH
                T1=TrialTime*(i-1) + sliding_window1*(window-1); % Burn-in period of 200 ms
                T2=T1+winsize1;   % Compute covariances until end of simulation
                
                PSTH_laserPairs_Laser(BlockTrial,window) = PSTH_laserPairs_Laser(BlockTrial,window) +...
                    histcounts(spikeTimes(spikeTimes>T1 & spikeTimes< T2)...
                    ,T1:winsize1:T2)/(winsize1*sum(StimulatedNeurons));
            end
        end
        
    end
    toc
end
% We summed spike counts over trials, now we divide by over how many trials
% we added them.
PSTH_laserPairs_Control = PSTH_laserPairs_Control / block_slide_size; 
PSTH_laserPairs_Laser = PSTH_laserPairs_Laser / block_slide_size;

%% We got a very noisy time series. So now we apply the Gaussian kernel.
PSTH_win = 30; %ms
sigma = PSTH_win; % std of gaussian kernel
sz = PSTH_win;    % length of gaussFilter vector
x = linspace(-sz / 2, sz / 2, sz);
gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
gaussFilter = gaussFilter / sum (gaussFilter);

% for i=1:NBlockTrials
%     PSTH_laserPairs_Laser(i,:) = conv(PSTH_laserPairs_Laser(i,:), gaussFilter,'same');
%     PSTH_laserPairs_Control(i,:) = conv(PSTH_laserPairs_Control(i,:), gaussFilter,'same');
% %     plot(linspace(0,1400,Number_of_points_PSTH),1000*PSTH_laserPairs_Laser(i,:))
% %     plot(linspace(0,1400,Number_of_points_PSTH),1000*filteredSignal)
% end

%%
% Calculate the mean rates for Control and Laser trials separately for
% each block.
reMeanControl = mean( reMean(1, ControlTrials) );
riMeanControl = mean( riMean(1, ControlTrials) );
reMeanLaser = mean( reMean( 1, LaserTrials ) ); % setdiff is used to only look at Laser trials.
riMeanLaser = mean( riMean( 1, LaserTrials ) ); % setdiff is used to only look at Laser trials.

reMeanControlLaserPeriod = mean( reLaserPeriod(1, ControlTrials) );
riMeanControlLaserPeriod = mean( riLaserPeriod(1, ControlTrials) );
reMeanLaserLaserPeriod = mean( reLaserPeriod( 1, LaserTrials ) ); % setdiff is used to only look at Laser trials.
riMeanLaserLaserPeriod = mean( riLaserPeriod( 1, LaserTrials ) ); % setdiff is used to only look at Laser trials.

reMeanControlPostLaser = mean( rePostLaserPeriod(1, ControlTrials) );
riMeanControlPostLaser = mean( riPostLaserPeriod(1, ControlTrials) );
reMeanLaserPostLaser = mean( rePostLaserPeriod( 1, LaserTrials ) ); % setdiff is used to only look at Laser trials.
riMeanLaserPostLaser = mean( riPostLaserPeriod( 1, LaserTrials ) ); % setdiff is used to only look at Laser trials.


% Mean rate over E and I pops for the whole block.
reMeanBlock=mean(reMean(1,:));
riMeanBlock=mean(riMean(1,:));
disp(sprintf('\nMean E and I rates from sims: %.2fHz %.2fHz',1000*reMeanBlock,1000*riMeanBlock))
disp(sprintf('\nLaser Mean E and I rates from sims: %.2fHz %.2fHz',1000*reMeanLaser,1000*riMeanLaser))
disp(sprintf('\nControl Mean E and I rates from sims: %.2fHz %.2fHz',1000*reMeanControl,1000*riMeanControl))

% Time-dependent mean rates
dtRate=100;
eRateTime=hist(s(1,s(2,:)<=Ne),1:dtRate:T)/(dtRate*Ne);
iRateTime=hist(s(1,s(2,:)>Ne),1:dtRate:T)/(dtRate*Ni);



TotalJee = JRec_ee*sqrt(N); % Record the whole time series of EE weights.
TotalJei = JRec_ei*sqrt(N); % Record the whole time series of EI weights.
TotalJie = JRec_ie*sqrt(N); % Record the whole time series of IE weights.
TotalJii = JRec_ii*sqrt(N); % Record the whole time series of II weights.

%     meanJei{BlockTrial} = mean(JRec)*sqrt(N);

%     AvgJei(BlockTrial) = mean( meanJei{BlockTrial}(:) );




%% Correlations for laser trials.
%%% All the code below computes spike count covariances and correlations

% Compute spike count covariances over windows of size
% winsize starting at time T1 and ending at time T2.

winsize=200; %ms.
sliding_window = 50; %ms.
Number_of_points_corr = (TrialTime/winsize-1)*(winsize/sliding_window); % Number of
% points in the time axis for which correlations are calculated.

block_slide_size = 20; % = how many trials in one block.
NBlockTrials = length(LaserTrials)-block_slide_size;

ComputeSpikeCountCorrs=1;
mReeLaser = zeros(NBlockTrials, Number_of_points_corr);
mReiLaser = zeros(NBlockTrials, Number_of_points_corr);
mRiiLaser = zeros(NBlockTrials, Number_of_points_corr);
mR_laserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);
mR_nonlaserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);
mR_stderr_laserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);
mR_stderr_nonlaserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);
mC_laserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);
mC_nonlaserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);
mC_stderr_laserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);
mC_stderr_nonlaserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);
Variance_laserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);
Variance_nonlaserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);
Variance_stderr_laserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);
Variance_stderr_nonlaserPairs_Laser = zeros(NBlockTrials, Number_of_points_corr);

corr_time = tic;

Control = 0; % 0 means calculate corrs for laser trials.
% 1 means calculate corrs for control trials.
parfor BlockTrial=1:NBlockTrials
    for window = 1:Number_of_points_corr
        T1=sliding_window*(window-1); % Burn-in period of 200 ms
        T2=T1+winsize;   % Compute covariances until end of simulation
        tic
        C=SpikeCountCovAriana(s,Ne*stimProb,T1,T2,winsize,1:Ne*stimProb,BlockTrial,...
            TrialTime,T,Control,ControlTrials,LaserTrials,block_slide_size);
        toc
        
%       Calculate covariances and variances separately to determine which causes
% the drop in correlations.

%         mC_laserPairs_Laser(BlockTrial,window)=mean( C(ismember(II, find(StimulatedNeurons==1)) & ismember(JJ, find(StimulatedNeurons==1)) & isfinite(C)) );
        colvect= C(~tril(ones(size(C))));
        mC_laserPairs_Laser(BlockTrial,window)= nanmean(colvect);
%         mC_stderr_laserPairs_Laser(BlockTrial,window)=std(C(ismember(II, find(StimulatedNeurons==1)) & ismember(JJ, find(StimulatedNeurons==1)) & isfinite(C))) ...
%             / sqrt((sum(StimulatedNeurons))) ;
        mC_stderr_laserPairs_Laser(BlockTrial,window)=nanstd(colvect)/ sqrt(Ne*stimProb);
        
        
%        mC_nonlaserPairs_Laser(BlockTrial,window)=mean(C(ismember(II, find(StimulatedNeurons==0)) & ismember(JJ, find(StimulatedNeurons==0)) & isfinite(C)));
%        mC_stderr_nonlaserPairs_Laser(BlockTrial,window)=std(C(ismember(II, find(StimulatedNeurons==0)) & ismember(JJ, find(StimulatedNeurons==0)) & isfinite(C)))...
%            / sqrt((Ne-sum(StimulatedNeurons)));
        
        Variance = diag(C);
        
%         Variance_laserPairs_Laser(BlockTrial,window)=mean( Variance( find(StimulatedNeurons==1) ) );
%         Variance_stderr_laserPairs_Laser(BlockTrial,window)=std(Variance( find(StimulatedNeurons==1)))/sqrt((sum(StimulatedNeurons))) ;
        Variance_laserPairs_Laser(BlockTrial,window)=nanmean(Variance);
        Variance_stderr_laserPairs_Laser(BlockTrial,window)=nanstd(Variance)/ sqrt(Ne*stimProb);
        
%        Variance_nonlaserPairs_Laser(BlockTrial,window)=mean(Variance( find(StimulatedNeurons==0) ) );
%        Variance_stderr_nonlaserPairs_Laser(BlockTrial,window)=std(Variance( find(StimulatedNeurons==0)))/sqrt((Ne-sum(StimulatedNeurons)));
        
        
        
        % Get correlation matrix from cov matrix
        if(ComputeSpikeCountCorrs)
            
            tic
            R=corrcov(C);
            toc
            %mReeLaser(BlockTrial,window)=mean(R(II<=Ne & JJ<=II & isfinite(R)));
            %mReiLaser(BlockTrial,window)=mean(R(II<=Ne & JJ>Ne & isfinite(R)));
            %mRiiLaser(BlockTrial,window)=mean(R(II>Ne & JJ>II & isfinite(R)));
            
            % Calculate correlations for laser pairs and non-laser pairs
            % separately like Ariana does.
            colvect= R(~tril(ones(size(R))));
            mR_laserPairs_Laser(BlockTrial,window)=nanmean(colvect);
            mR_stderr_laserPairs_Laser(BlockTrial,window)=nanstd(colvect)/ sqrt(Ne*stimProb);
            
%             mR_laserPairs_Laser(BlockTrial,window)=mean( R(ismember(II, find(StimulatedNeurons==1)) & ismember(JJ, find(StimulatedNeurons==1)) & isfinite(R)) );
%             mR_stderr_laserPairs_Laser(BlockTrial,window)=std(R(ismember(II, find(StimulatedNeurons==1)) & ismember(JJ, find(StimulatedNeurons==1)) & isfinite(R))) ...
%                 / sqrt((sum(StimulatedNeurons))) ;
                        
%            mR_nonlaserPairs_Laser(BlockTrial,window)=mean(R(ismember(II, find(StimulatedNeurons==0)) & ismember(JJ, find(StimulatedNeurons==0)) & isfinite(R)));
%            mR_stderr_nonlaserPairs_Laser(BlockTrial,window)=std(R(ismember(II, find(StimulatedNeurons==0)) & ismember(JJ, find(StimulatedNeurons==0)) & isfinite(R)))...
%                / sqrt((Ne-sum(StimulatedNeurons)));
            
            % Mean-field spike count correlation matrix
            %mR{BlockTrial,window}=[mRee mRei; mRei mRii];
        end
    end
end


disp('\n Computation of laser correlations finished.')
toc(corr_time)/3600

%% Correlations for Control trials. 
%%% All the code below computes spike count covariances and correlations

% Compute spike count covariances over windows of size
% winsize starting at time T1 and ending at time T2.
winsize=200;

ComputeSpikeCountCorrs=1;
mReeControl = zeros(NBlockTrials, Number_of_points_corr);
mReiControl = zeros(NBlockTrials, Number_of_points_corr);
mRiiControl = zeros(NBlockTrials, Number_of_points_corr);
mR_laserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);
mR_nonlaserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);
mR_stderr_laserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);
mR_stderr_nonlaserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);
mC_laserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);
mC_nonlaserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);
mC_stderr_laserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);
mC_stderr_nonlaserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);
Variance_laserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);
Variance_nonlaserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);
Variance_stderr_laserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);
Variance_stderr_nonlaserPairs_Control = zeros(NBlockTrials, Number_of_points_corr);

Control = 1; % 0 means calculate corrs for laser trials.
% 1 means calculate corrs for control trials.

corr_time = tic;

% Make it optional, in case we only want to look at Laser trials only.
if(Control)
    parfor BlockTrial=1:NBlockTrials
        for window = 1:Number_of_points_corr
            T1=sliding_window*(window-1); % Burn-in period of 250 ms
            T2=T1+winsize;   % Compute covariances until end of simulation
            tic
            C=SpikeCountCovAriana(s,Ne*stimProb,T1,T2,winsize,1:Ne*stimProb,BlockTrial,...
            TrialTime,T,Control,ControlTrials,LaserTrials,block_slide_size);
            toc
            
            colvect= C(~tril(ones(size(C))));
            mC_laserPairs_Control(BlockTrial,window)=nanmean(colvect);
            mC_stderr_laserPairs_Control(BlockTrial,window)=nanstd(colvect)/ sqrt(Ne*stimProb);
            
            
%             mC_laserPairs_Control(BlockTrial,window)=mean( C(ismember(II, find(StimulatedNeurons==1)) & ismember(JJ, find(StimulatedNeurons==1)) & isfinite(C)) );
%             mC_stderr_laserPairs_Control(BlockTrial,window)=std(C(ismember(II, find(StimulatedNeurons==1)) & ismember(JJ, find(StimulatedNeurons==1)) & isfinite(C))) ...
%                 / sqrt((sum(StimulatedNeurons))) ;
            
%            mC_nonlaserPairs_Control(BlockTrial,window)=mean(C(ismember(II, find(StimulatedNeurons==0)) & ismember(JJ, find(StimulatedNeurons==0)) & isfinite(C)));
%            mC_stderr_nonlaserPairs_Control(BlockTrial,window)=std(C(ismember(II, find(StimulatedNeurons==0)) & ismember(JJ, find(StimulatedNeurons==0)) & isfinite(C)))...
%                / sqrt((Ne-sum(StimulatedNeurons)));
            
            Variance = diag(C);
            Variance_laserPairs_Control(BlockTrial,window)=nanmean(Variance);
            Variance_stderr_laserPairs_Control(BlockTrial,window)=nanstd(Variance)/sqrt(Ne*stimProb);
            
%             Variance_laserPairs_Control(BlockTrial,window)=mean( Variance( find(StimulatedNeurons==1) ) );
%             Variance_stderr_laserPairs_Control(BlockTrial,window)=std(Variance( find(StimulatedNeurons==1)))/sqrt((sum(StimulatedNeurons))) ;
            
%            Variance_nonlaserPairs_Control(BlockTrial,window)=mean(Variance( find(StimulatedNeurons==0) ) );
%            Variance_stderr_nonlaserPairs_Control(BlockTrial,window)=std(Variance( find(StimulatedNeurons==0)))/sqrt((Ne-sum(StimulatedNeurons)));
            
            % Get correlation matrix from cov matrix
            if(ComputeSpikeCountCorrs)
                
                tic
                R=corrcov(C);
                toc
                %mReeControl(BlockTrial,window)=mean(R(II<=Ne & JJ<=II & isfinite(R)));
                %mReiControl(BlockTrial,window)=mean(R(II<=Ne & JJ>Ne & isfinite(R)));
                %mRiiControl(BlockTrial,window)=mean(R(II>Ne & JJ>II & isfinite(R)));
                
                colvect= R(~tril(ones(size(R))));
                mR_laserPairs_Control(BlockTrial,window)=nanmean(colvect);
                mR_stderr_laserPairs_Control(BlockTrial,window)=nanstd(colvect)/sqrt(Ne*stimProb);
                
                % Calculate correlations for laser pairs and non-laser pairs
                % separately like Ariana does.
%                 mR_laserPairs_Control(BlockTrial,window)=mean(R(ismember(II, find(StimulatedNeurons==1)) & ismember(JJ, find(StimulatedNeurons==1)) & isfinite(R)));
%                 mR_stderr_laserPairs_Control(BlockTrial,window)=std(R(ismember(II, find(StimulatedNeurons==1)) & ismember(JJ, find(StimulatedNeurons==1)) & isfinite(R))) ...
%                     / sqrt((sum(StimulatedNeurons))) ;
                
%                mR_nonlaserPairs_Control(BlockTrial,window)=mean(R(ismember(II, find(StimulatedNeurons==0)) & ismember(JJ, find(StimulatedNeurons==0)) & isfinite(R)));
%                mR_stderr_nonlaserPairs_Control(BlockTrial,window)=std(R(ismember(II, find(StimulatedNeurons==0)) & ismember(JJ, find(StimulatedNeurons==0)) & isfinite(R)))...
%                    / sqrt((Ne-sum(StimulatedNeurons)));
               
                
                % Mean-field spike count correlation matrix
                %mR{BlockTrial,window}=[mRee mRei; mRei mRii];
            end
        end
    end
end
toc(corr_time)/3600


%% Save some variables for plots.
spikeTimes = s( 1 ,s(2,:)>3900 & s(2,:)<4100 ); % Half E and Half I.
spikeIndex = s( 2 ,s(2,:)>3900 & s(2,:)<4100 );
%spikeTimes = spikeTimes(1:50000);
%spikeIndex = spikeIndex(1:50000);

% spikeTimes = s{1}(1, ismember(s{1}(2,:),find(StimulatedNeurons==1)));
% spikeIndex = s{1}(2, ismember(s{1}(2,:),find(StimulatedNeurons==1)));


%%
mRLaserPeriod = zeros(NBlockTrials,1);
mRPostLaserPeriod = zeros(NBlockTrials,1);
laserOnSet1 = laserOnSet - winsize;

for i=1:NBlockTrials
    mRLaserPeriod(i) = mean( (mReeLaser(i,laserOnSet1/sliding_window:(laserOnSet1+StimDuration)/sliding_window)+2*mReiLaser(i,laserOnSet1/sliding_window:(laserOnSet1+StimDuration)/sliding_window)+mRiiLaser(i,laserOnSet1/sliding_window:(laserOnSet1+StimDuration)/sliding_window) )/4 );  
    mRPostLaserPeriod(i) = mean( (mReeLaser(i,(laserOnSet1+StimDuration)/sliding_window:(laserOnSet1+StimDuration+300)/sliding_window)+2*mReiLaser(i,(laserOnSet1+StimDuration)/sliding_window:(laserOnSet1+StimDuration+300)/sliding_window)+mRiiLaser(i,(laserOnSet1+StimDuration)/sliding_window:(laserOnSet1+StimDuration+300)/sliding_window))/4 );
end
% 450ms above is the time we see in the post laser period.

toc(start_time)/3600


%% Save variables needed for plots in "ArianaVariables.mat".
save( ['/scratch/AlanAkil/ArianaVariables_ee_ei_ie_ii_seed_' num2str(seed) '_scale_' num2str(scaling_factor) '_freq_100.mat'], 'seed',...
    'NBlockTrials','eta_ee','eta_ei','eta_ie_hebbian','scaling_factor','laser_freq',...
    'eta_ie_homeo','eta_ii','Jmax_ee','Jmax_ie','winsize','T','Total_number_trials',...
    'ControlTrials','block_slide_size','LaserTrials',...
    'Number_of_points_corr','sliding_window','TotalJei','TotalJee','TotalJie','TotalJii',...
    'dt','N','TrialTime','Jm',...
    'laserOnSet', 'mR_laserPairs_Control',...
    'mR_laserPairs_Laser', ...
    'mR_stderr_laserPairs_Control','mR_stderr_laserPairs_Laser',...
    'mC_laserPairs_Control','mC_laserPairs_Laser',...
    'mC_stderr_laserPairs_Control','mC_stderr_laserPairs_Laser',...
    'Variance_laserPairs_Control',...
    'Variance_laserPairs_Laser',...
    'Variance_stderr_laserPairs_Control',...
    'Variance_stderr_laserPairs_Laser',...
    'FR_laserPairs_Laser', 'FR_laserPairs_Control',...
    'PSTH_laserPairs_Laser','PSTH_laserPairs_Control','PSTH_win',...
    'Number_of_points_PSTH',...
    'IeCurrent','IiCurrent','IxCurrent','IiCurrent_stim')

%% Note when the sims have finished in the log file.
fprintf('Simulation has finished.\n')

end
