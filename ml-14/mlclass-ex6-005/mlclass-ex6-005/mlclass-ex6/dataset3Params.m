function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

step_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10];
%x1 = [1 2 1]; x2 = [0 4 -1];

mean_value = 1;

for i = 1 : length(step_vec)
	for j = 1 : length(step_vec)
		C_i = step_vec(i);
		sigma_j = step_vec(j);
		model= svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j));
		predictions = svmPredict(model, Xval);
		%fprintf('computed mean is : %f\n', mean(double(predictions ~= yval)));
		if mean_value > mean(double(predictions ~= yval))
			mean_value = mean(double(predictions ~= yval));
			C = C_i;
			sigma = sigma_j;
			%fprintf('the smallest mean_value is : %f, C is %f, sigma is %f\n', mean_value, C, sigma);
		end
		
	end
end





% =========================================================================

end
