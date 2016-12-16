% script for nltcs data example that appears in the paper

clear; 

% if ismac
%     addpath('~/Dropbox (Personal)/utilities/Matlab Functions/Graphical');
%     addpath('~/Dropbox (Personal)/utilities/Matlab Functions/Tensor');
%     addpath('~/Dropbox (Personal)/utilities/Matlab Functions');
% elseif ispc
%     addpath('c:/Users/johndrow/Dropbox/utilities/Matlab Functions/Graphical');
%     addpath('c:/Users/johndrow/Dropbox/utilities/Matlab Functions/Tensor');
%     addpath('c:/Users/johndrow/Dropbox/utilities/Matlab Functions');
% elseif isunix
%     addpath('/home/james/Dropbox/utilities/Matlab Functions/Graphical');
%     addpath('/home/james/Dropbox/utilities/Matlab Functions/Tensor');
%     addpath('/home/james/Dropbox/utilities/Matlab Functions');
% end


p=16; d = 2^p; d0 = 2;
a = .5;
makedesign = false;
saving = false;

if makedesign
    [X,~,polyt,ord] = llmdesign2(p,d0,true);
else
    load(strcat('Outputs/lldesign_',num2str(p),'.mat'));
end

X0 = X; X0 = X0(:,sum(X0,1)==2^(p-1)); X0 = full(X0);
X0f = X0;
X = X(2:end,2:end);
Xf = X;
dat = readtable('NLTCS/nltcs.txt','delimiter','tab','readvariablenames',false);
N = size(dat,1);

dat = table2array(dat(:,2:end));

y = zeros(d,1);
for j=1:d
    tmp = all(bsxfun(@eq,X0f(j,:),dat),2);
    y(j) = sum(tmp);
end

iX = X\speye(d-1);
% posterior
lam = y+a;    

mu = psi(lam(2:end))-psi(lam(1));
sigma = psi(1,lam(2:end)) + psi(1,lam(1));
muhm = iX*mu;

tmp = zeros(d-1,1); psitmp = psi(1,lam(2:end));
for j=1:(d-1)
    tmp(j) = (psitmp'.*iX(j,:))*iX(:,j);
end

sigmahm = tmp + psi(1,lam(1)); sigmahm = sqrt(sigmahm);

if saving
    save('Outputs/nltcs_full_post.mat');
    save('Outputs/nltcs_full_small_post.mat','muhm','sigmahm','y','X0');
end


