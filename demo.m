function  demo()


X = csvread('X.csv');
y = csvread('Y.csv');
m = length(y);
fprintf('First 10 examples from the dataset: \n');
fprintf(' X goes here ============');
X
pause

fprintf(' Y goes here ============');
y

pause

fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);

pause
        
        % Add intercept term to X
        X = [ones(m, 1) X];

pause

        fprintf('Running gradient descent ...\n');
        
        % Choose some alpha value
        alpha = 0.01;
        num_iters = 400;
        
        % Init Theta and Run Gradient Descent
        theta = zeros(7, 1);
        [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
pause
        
        % Plot the convergence graph
        figure;
        plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
        xlabel('Number of iterations');
        ylabel('Cost J');
        
        % Display gradient descent's result
        fprintf('Theta computed from gradient descent: \n');
        fprintf(' %f \n', theta);
        fprintf('\n');

pause
        
        input = [13,5,3,16,4100,9999];
        
        normalized = (input-mu)./sigma;

        prediction = [1,normalized]*theta;

 fprintf('Theta computed from gradient descent: \n');
prediction

        end
