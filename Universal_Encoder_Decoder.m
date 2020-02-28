% Thomas Merkh, July 13th 2017, tmerkh@ucla.edu
% Near Optimal Signal Recovery From Random Projections - Ref Terence Tao
% Implementation of section 8 - 'Universal Encoding'

% In this program, a sparse random signal, f, of length n is created.  Then, m measurements are taken, where
% m << n, and solving an l1 optimization problem, otherwise known as the 'basis pursuit' problem, recovers the signal
% with great precision.  The basis pursuit problem is minimize ||g||_1 for g \in \mathbb{R^n} subject to Xg = y, where
% y is the compressed data transmitted (Xf = y \in \mathbb{R^m}).  Upon solving this, using the open-source solver YALL1,
% credits to RICE university - see http://yall1.blogs.rice.edu/ for details - the original signal is plotted against the 
% recovered one, and the relative error is measured.

function Universal_Encoder_Decoder()

	n = 1000;										%Cardinality of signal
	m = 300;										%Number of measurements (less than n, or else its a trivial problem)
	k = 70;											%Size of support of signal f
	
	[X,y,f] = gen_data(m,n,k,0.00,0);				%Auxilary function written to generate a sparse signal and random measurements
	
	% call solver
	opts.maxit = 10000;
	opts.tol = 5*10^(-6);
	opts.print = 0;
	opts.nonorth = 1; 								%because X'*X neq I, else 0
	disp('YALL1: Basis Pursuit');
	opts.nu = 0;
	opts.rho = 0;
	tic; [ff,Out] = yall1(X,y,opts); toc
	rerr = norm(ff-f)/norm(f);
	fprintf('Iter %4i: Rel_err = %6.2e\n\n',Out.iter,rerr);

	% Plot comparison figure
	figure1 = figure('Name','l1 Sparse Signal Recovery');
	axes1 = axes('Parent',figure1);
	hold(axes1,'on');
	plot(linspace(1,n,n),f,'Parent',axes1, 'LineWidth', 2, 'Color', [0 0 0]);
	plot(linspace(1,n,n),ff,'Parent',axes1, 'LineWidth', 1, 'Color', [1 0 0]);
	legend('True Signal','Recovered Signal');

	xlabel('Time');
	ylabel('Signal Strength');
	title('l1 Sparse Signal Recovery')

endfunction


function [X,y,f] = gen_data(m,n,k,sigma,perc)
	X = randn(m,n);
	d = 1./sqrt(sum(X.^2));
	X = X*sparse(1:n,1:n,d);
	f = zeros(n,1);
	p = randperm(n);
	f(p(1:k)) = randn(k,1);
	y = X*f;
	
	% white noise
	white = sigma*randn(m,1);
	y = y + white;
	% impulsive noise
	p = randperm(m);
	L = floor(m*perc/100);
	y(p(1:L)) = 2*(rand(L,1) > 0) - 1;

endfunction;