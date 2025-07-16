% Step 1: Import dataset (assume CSV with headers)
filename = 'weather_data.csv';  % Replace with your file name
data = readtable(filename);

% Step 2: Convert necessary columns to categorical
data.Outlook = categorical(data.Outlook);
data.Humidity = categorical(data.Humidity);
data.Wind = categorical(data.Wind);
data.Run = categorical(data.Run);

% Step 3: Build Naive Bayes classifier
% Predictor variables
predictors = data(:, {'Outlook', 'Humidity', 'Wind'});
% Target variable
response = data.Run;

% Train the model
model = fitcnb(predictors, response);

% Step 4: Create new test data
newData = table(categorical("Rainy"), categorical("Normal"), categorical("Weak"),'VariableNames', {'Outlook','Humidity','Wind'});

% Predict
prediction = predict(model, newData);
fprintf('Prediction: Run = %s\n', string(prediction));

% Step 5: Predict on training data and evaluate
train_pred = predict(model, predictors);
confusionchart(response, train_pred);  % Shows classification performance visually

% Accuracy
accuracy = sum(train_pred == response) / numel(response);
fprintf('Training Accuracy: %.2f%%\n', accuracy * 100);
