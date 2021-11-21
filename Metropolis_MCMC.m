% Metropolis Markov Chain Monte Carlo

% Initializing parameters
T = 10000;   % Number of iterations. 

%%%%% Initialization of chain sampling from target distribution prior %%%%%
mu = [0 0];                     % Means vector
rho = 0.998;                    % Covariance
sigma = [1 rho; rho 1];         % Covariance matrix of target distribution
P = @(X) mvnpdf(X, mu, sigma);  % Target probability distribution function

%%%%%%%%%%%%%%%%%%%%%%%%% Proposal Distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set variance of the distribution to the smallest variance of the target %
% distribution.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Eigenvector of the target distribution.
proposal_eigenvector = eig(sigma);
% Extract the smallest variance and construct the diagonal covariance
% matrix.
proposal_covariance = proposal_eigenvector(1) * eye(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%% Metropolis Setup  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = mu;                    % Initialise the current/first step.
step_stored = zeros(T, 2); % Matrix to store random walk steps. 
step_stored(1,:) = x;      % Initialize the matrix with the current step.
accepted = 0;              % Total steps that are accepted.

% Function to randomly sample from the proposal distribution.
Q_sample = @(x) mvnrnd(x, proposal_covariance);
% Function to calculate the A value.
A = @(x_prime, x) P(x_prime)/P(x);

%%%%%%%%%%%%%%%%%%%%%%%%% Metropolis Algorithm  %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Random Walk by iterating over the steps.
for i=1:T-1
   % Proposed new step from proposal distribution with mean of p.
   x_prime  = Q_sample(x);
 
   % Evaluate the target distribution of the proposed step.
   A_value =  A(x_prime,x);
   
   % Evaluate the acceptance of the proposed step.
   if A_value >= 1
       accept = 1;
   elseif A_value > rand()
       accept = 1;
   else
       accept = 0;
   end
   
   % If the proposed step is accepted then update the current step to the
   % proposed step.
   if accept
       x = x_prime;
   end
   % Increment the accepted steps.
   accepted = accepted + accept;
   % Update the stored steps matrix with the update current step.
   step_stored(i+1,:) = x;
end
   
% Calculate the acceptance rate
acceptance_rate = accepted/(T-1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Metropolis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Contour Plot of target distribution
x1 = linspace(-1, 1);
x2 = x1;
[x1 x2] = meshgrid(x1, x2);
Pcontour = reshape(P([x1(:), x2(:)]), 100, 100);
contour(x1, x2, Pcontour, [1.6 3], 'k'); 
hold on
% Iterate over the sample and draw the random walk.
for i=0:length(x)
  plot(step_stored(:,1),step_stored(:,2) , 'b.-','LineWidth', 1, 'MarkerSize', 4)
end
axis square;
hold off
