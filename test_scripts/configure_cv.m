function [train, test, n_folds] = configure_cv(Y, fold, varargin)

% varargin = {cvmode,[n_folds]}

try 
    cvmode = varargin{1};
catch
    cvmode = 'loo2';
end

n_subjects = size(Y,1);
n_classes  = size(Y,2);

train = []; test = [];
switch cvmode
    case 'loo1' % basic LOO-CV
        for t = 1:n_subjects
           if t == fold
               test = [test t];
           else
               train = [train t];
           end
        end
        
    case 'loo2' % LOO-CV holding out one subject per class
        
        n_folds = max(sum(Y));
        for c = 1:n_classes
            class{c} = find(Y(:,c))';
        end
        
        for c = 1:n_classes
            for f = 1:n_folds
                if f <= length(class{c})
                    if f == fold
                        test = [test class{c}(f)];
                    else
                        train = [train class{c}(f)];
                    end
                end
            end
        end
        
    case 'kfold'
        try 
            n_folds = varargin{2};
        catch
            disp('k-fold CV specified without n_folds, defaulting to 4');
            n_folds = 4;
        end
        
        % Configure training and test indices for cross-validation
        for c = 1:n_classes
            fold_size{c} = floor(length(find(Y(:,c))) / n_folds);
            class{c} = find(Y(:,c))';
        end
        train = []; test = [];
        
        for c = 1:n_classes
            for tf = 1:n_folds
                if tf == n_folds
                    idx = (tf - 1)*fold_size{c}+1:length(find(Y(:,c)));
                else
                    idx = (tf - 1)*fold_size{c} + (1:fold_size{c});
                end
                
                if fold == tf
                    test = [test class{c}(idx)];
                else
                    train = [train class{c}(idx)];
                end
            end
        end
        
    otherwise
        error ('Invalid CV mode specified');
end
