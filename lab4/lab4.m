%% lab4.m
% 
% clear; clc; close all;
 addpath Qlearning;
%% Random path choosing
% 
% state = gwaction(1); % init random state
% state.isterminal = 0; % Make sure not in goal
% gwinit(1)
% gwdraw;shg;
% 
% while (state.isterminal ~= 1)
%     action = sample([1 2 3 4], [0.2 0.1 0.1 0.6]); % Random action choosing
%     state = gwaction(action);
%     gwplotarrow(state.pos, action);
%     pause(0.1);
% end
% 
%% Q-learning with epsilon-greedy

world = 3; % World to explore

% Init Q-function
xSize = 10; ySize = 15; aSize = 4; aDim = 3;
Q = ones(xSize, ySize, aSize); % Init with ones
%Q = rand(xSize, ySize, aSize); % Init with random numbers
% prevent invalid actions on world 4
if world == 4
    Q(end,:,1)=-Inf;
    Q(1,:,2)=-Inf;
    Q(:,end,3)=-Inf;
    Q(:,1,4)=-Inf;
end

% Parameters
alpha = 0.2; gamma = 0.1; epsilon = 0.1; epsilonmin = 0.1;
numEpisodes = 1000;
plotProgress = 1;
%%%%%%%%%%%%%%%%%%%%%
%% Start Q-learning %
%%%%%%%%%%%%%%%%%%%%%
%profile on;
% updated Q = (1-alpha) * prevQ + alpha * (r + gamma *  max(Q, [], aDim))

% Q-learning algorithm
disp(['Starting Q-learning on world ', num2str(world),','])
disp(['using epsilon greedy strategy.'])
disp(['epsilon = ',num2str(epsilon),'. Number of iterations: ',num2str(numEpisodes)]);
tic;
S = '[--------------------]';
tmp = 0;


for episode = 1:numEpisodes % for each episode
    % show percentual progress
    if mod(episode,numEpisodes/100)==0
        tmp = tmp+1;
        if mod(tmp,5)==0
            S(tmp/5 + 1) = '%';
        end
        disp([S,num2str((tmp)),' % done']);
    end;
    
    % init a start state s
    gwinit(world)
    % draw episode
    if plotProgress
        gwdraw;
    end
    
    state = gwstate;
    
    % loop until goal is found
    while (state.isterminal ~= 1) % for each step k in episode
        
        % Choose action
        greedy = sample([0 1], [epsilon (1-epsilon)]);
        if (greedy == 0)
            action = sample([1 2 3 4], [0.25 0.25 0.25 0.25]); % random action
        else
            [~,~,a] = find(max(Q(state.pos(1),state.pos(2),:), [], aDim));
            action = find(Q(state.pos(1),state.pos(2),:) == a);
            % if same Q-val for different actions, choose randomly
            if length(action)>1
                action = sample(action, ones(size(action))/length(action));
            end 
        end
        
        % Take action
        nextState = gwaction(action);
        
        % check if action is valid, else choose randomly
        while (nextState.isvalid ~= 1)
            % update Q-function to -INF for forbidden actions except for
            % world 4
            if world~=4
                Q(nextState.pos(1),nextState.pos(2),action) = -Inf;
            end
            % choose new random action
            action = sample([1 2 3 4], [0.25 0.25 0.25 0.25]);
            nextState = gwaction(action);
        end
        
        % check if goal reached
        if (nextState.isterminal == 1)
            Q(nextState.pos(1),nextState.pos(2),:) = 0; % All Q-values in goal is 0
        else
            % Observe reward r and next state s_(k+1) HOW NEXT STATE CARE?
            r = nextState.feedback;
            
            % Update estimated Q-function
            Q(state.pos(1),state.pos(2),action) = (1-alpha) * Q(state.pos(1),state.pos(2),action) + alpha * ...
                (r + gamma *  max(Q(nextState.pos(1),nextState.pos(2),:), [], aDim));
        end
        
        % plot robots path to goal
        if plotProgress
            gwplotarrow(state.pos, action);
            pause(0.001);
        end
        state = nextState;
    end

    % update epsilon, may want to explore more in first episodes
    epsilon = max(epsilon*0.99,epsilonmin);
end
disp('Done');
elapt = toc;
disp(['Elapsed time: ', num2str(uint32(elapt)),' s.']);
disp(['Average time per iteration: ' num2str(elapt/numEpisodes),' s.']);

%%%%%%%%%%%%%%%%%%%%%
%% Plot V*-function %
%%%%%%%%%%%%%%%%%%%%%
%close all;
figure(1);
gwdraw;
[V_star,a_star] = max(Q,[],aDim);
x = 1:size(V_star,2);
y = 1:size(V_star,1);
for i = x
    for j = y
        gwplotarrow([y(j);x(i)],a_star(j,i))
    end
end
title 'Estimated optimal policy'
figure(2);
imagesc(V_star); title 'V*'; axis image; xlabel Y; ylabel X; colorbar;