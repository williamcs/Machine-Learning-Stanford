data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

%function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples


for iter = 1:iterations

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    theta(1) = theta(1) - alpha*(1/m)*sum((X*theta - y).* X(:, 1))
    theta(2) = theta(2) - alpha*(1/m)*sum((X*theta - y).* X(:, 2))
    
%     grad = (1/m).* X' * ((X * theta) - y);
%     theta = theta - alpha .* grad
    
%     temp = (X * theta - y)
%     grad1 = sum(temp .* X(:, 1))
%     grad2 = sum(temp .* X(:, 2))
%     
%     theta(1) = theta(1) - alpha/m * grad1
%     theta(2) = theta(2) - alpha/m * grad2
    





    % ============================================================

end

% print theta to screen
theta




















