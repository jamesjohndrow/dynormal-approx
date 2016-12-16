function [ X,Xv,polyt,ord ] = llmdesign2(p,d,fulldesign)
%LLMDESIGN2 creates a design matrix for a log-linear model
%   for now, # of levels must be same for all vars


Xv = repmat((1:d)',[d^(p-1) 1]);
for j=1:p-1
   Xv = horzcat(Xv,repmat(sortrows(repmat((1:d)',[d^j 1])),[d^(p-1-j) 1])); 
end

% indicators for the main effects, use to make full dummy/binary matrix

if fulldesign
    X = dummyvar(Xv);
    Xnew = sparse(ones(d^p,1));
else
    X = []; 
end
ord = zeros(d^p,1);
polyt = struct();
polyt(j).vars = [];
polyt(j).levels = ones(d,1);
for j=2:d^p    
    polyt(j).vars = find(Xv(j,:)>1);
    polyt(j).levels = Xv(j,polyt(j).vars);
    ord(j) = length(polyt(j).vars);
    Xidx = (polyt(j).vars-1)*d+polyt(j).levels;
    if fulldesign 
        Xnew = horzcat(Xnew,sparse(prod(X(:,Xidx),2)));
    end
end
%Xv = X;
if fulldesign
    X = Xnew;
end

end

