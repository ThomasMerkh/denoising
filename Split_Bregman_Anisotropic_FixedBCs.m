% Thomas Merkh, tmerkh@ucla.edu, Last edit: July 19th, 2017
% Ref: "The Split Bregman Method for L1 Regularized Prroblems", Tom Goldstein, Stanley Osher
% This method solves a wide variety of constrainted optimization problems, specifically constrainted
% l1 regularized problems, i.e. min_u ||Phi(u)||_l1 + H(u) where those are convex functions.
% Examples: TV/ROF denoising and Basis Pursuit problems.
% This Algorithm is on page 10 of the referenced paper, 
% and the parameters used are the same as given in section 5, numerical results

clear;

function GG = grad(y, dirr, i, j)
	N = size(y,2);
	if(dirr == 1) % x direction
		if(i == 1)
			GG = y(N,j) - y(i+1,j);
		elseif(i == N)
			GG = y(i-1,j) - y(1,j);
		else
			GG = y(i-1,j) - y(i+1,j);
		end
	elseif(dirr == 2) % y direction
		if(j == 1)
			GG = y(i,N) - y(i,j+1);
		elseif(j == N)
			GG = y(i,j-1) - y(i,1);
		else
			GG = y(i,j-1) - y(i,j+1);
		end
	else
		GG = 0;
		disp('Error in taking Gradient!')
	end
	GG = GG/2.0;
endfunction

function s = shrink(y,b,gamma, dirr)
	if(dirr == 1) % x direction
		for i = 2:size(y,1)-1
			for j = 2:size(y,2)-1
				x(i,j) = b(i,j) + grad(y,1,i,j);
			end
		end
		s = x/norm(x,1)*max([norm(x,1) - gamma, 0]);

	elseif(dirr == 2) % y direction
		for i = 2:size(y,1)-1
			for j = 2:size(y,2)-1
				x(i,j) = b(i,j) + grad(y,2,i,j);
			end
		end
		s = x/norm(x,1)*max([norm(x,1) - gamma, 0]);
	else
		s = zeros(size(y,2));
	end
endfunction

% Generate Test Data - Assuming fixed BCS - Whatever they start as, they stay as
N = 128+2;											%Size of system
Cut = 2;											%Makes Jumps in initial data: 1 == Horizontal, 2 == Diagonal
k = floor(N*0.4); 									%How much noise to add
noise_strength = 0.0;
noise = zeros(N);
noise(randi([1 N], 1, k),randi([1 N], 1, k) ) = randn(k);
f_gaussian = exp(-(ones(1,N)'*linspace(-3,3,N)).^2) + noise_strength.*noise;
f = zeros(N);
%Gaussian_random_numbers = sqrt(2)*erfinv(rand(N))
f((rand(N) < f_gaussian)) = 1;

if(Cut == 1)
	cutset = zeros(N);
	cutset(1:floor(N/2),:) = 1;
	f(cutset > 0) = f(cutset > 0)*2;
elseif(Cut == 2)
	for i = 1:N
		for j = 1:N
			if(i+j > N)
				f(i,j) = f(i,j)*2;
			end
		end
	end
end
	

% Plot original signal
figure1 = figure('Name','Split Bregman Anisotropic TV Denoising');
axes1 = axes('Parent',figure1);
hold(axes1,'on');
xlim([2 N-1]);
ylim([2 N-1]);
mesh(f,'Parent',axes1);

xlabel('x-Signal');
ylabel('y-Signal');
zlabel('Signal Strength')
title('Noisy Image or Data');
legend('Original Signal');

u = f;
u_new = f;
u_to_update = ones(N);
dx = zeros(N); bx = zeros(N);
dy = zeros(N); by = zeros(N);
lambda = 0.1;								%These parameters may be optimized by minimizing the conditioning number, lambda = 0.1
mu = 0.05;									%These parameters may be optimized by minimizing the conditioning number, usually lambda = 2mu
tol = 0.005; 								%tolerance 
iteration = 0;
itermax = 200;

while(norm(u - u_to_update)/norm(u_to_update) > tol && iteration < itermax)

	%interior
	for i = 3:N-2
		for j = 3:N-2
			u_new(i,j) = (lambda/(mu + 4*lambda))*(u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1) + ...
				+ dx(i-1,j) - dx(i,j) + dy(i,j-1) - dy(i,j) + ...
				+ bx(i-1,j) - bx(i,j) + by(i,j-1) - by(i,j) ) + (mu/(mu+4*lambda))*f(i,j);
		end
	end

	%edges
	for j = 3:N-2
		u_new(2,j) = (lambda/(mu + 4*lambda))*(u(3,j) + u(1,j) + u(2,j+1) + u(2,j-1) + ...
				+ dx(1,j) - dx(2,j) + dy(2,j-1) - dy(2,j) + ...
				+ bx(1,j) - bx(2,j) + by(2,j-1) - by(2,j) ) + (mu/(mu+4*lambda))*f(1,j);
		u_new(N-1,j) = (lambda/(mu + 4*lambda))*(u(N,j) + u(N-2,j) + u(N-1,j+1) + u(N-1,j-1) + ...
				+ dx(N-2,j) - dx(N-1,j) + dy(N-1,j-1) - dy(N-1,j) + ...
				+ bx(N-2,j) - bx(N-1,j) + by(N-1,j-1) - by(N-1,j) ) + (mu/(mu+4*lambda))*f(N-1,j);
	end
	for i = 3:N-2
		u_new(i,2) = (lambda/(mu + 4*lambda))*(u(i+1,2) + u(i-1,2) + u(i,3) + u(i,1) + ...
				+ dx(i-1,2) - dx(i,2) + dy(i,1) - dy(i,2) + ...
				+ bx(i-1,2) - bx(i,2) + by(i,1) - by(i,2) ) + (mu/(mu+4*lambda))*f(i,2);
		u_new(i,N-1) = (lambda/(mu + 4*lambda))*(u(i+1,N-1) + u(i-1,N-1) + u(i,N) + u(i,N-2) + ...
				+ dx(i-1,N-1) - dx(i,N-1) + dy(i,N-2) - dy(i,N-1) + ...
				+ bx(i-1,N-1) - bx(i,N-1) + by(i,N-2) - by(i,N-1) ) + (mu/(mu+4*lambda))*f(i,N-1);
	end

	%corners
	u_new(2,2) = (lambda/(mu + 4*lambda))*(u(3,2) + u(1,2) + u(2,3) + u(2,1) + ...
				+ dx(1,2) - dx(2,2) + dy(2,1) - dy(2,2) + ...
				+ bx(1,2) - bx(2,2) + by(2,1) - by(2,2) ) + (mu/(mu+4*lambda))*f(2,2);
	u_new(2,N-1) = (lambda/(mu + 4*lambda))*(u(3,N-1) + u(1,N-1) + u(2,N) + u(2,N-2) + ...
				+ dx(1,N-1) - dx(2,N-1) + dy(2,N-2) - dy(2,N-1) + ...
				+ bx(1,N-1) - bx(2,N-1) + by(2,N-2) - by(2,N-1) ) + (mu/(mu+4*lambda))*f(2,N-1);
	u_new(N-1,2) = (lambda/(mu + 4*lambda))*(u(N,2) + u(N-2,2) + u(N-1,3) + u(N-1,1) + ...
				+ dx(N-2,2) - dx(N-1,2) + dy(N-1,1) - dy(N-1,2) + ...
				+ bx(N-2,2) - bx(N-1,2) + by(N-1,1) - by(N-1,2) ) + (mu/(mu+4*lambda))*f(N-1,2);
	u_new(N-1,N-1) = (lambda/(mu + 4*lambda))*(u(N,N-1) + u(N-1-1,N-1) + u(N-1,N) + u(N-1,N-1-1) + ...
			+ dx(N-1-1,N-1) - dx(N-1,N-1) + dy(N-1,N-1-1) - dy(N-1,N-1) + ...
			+ bx(N-1-1,N-1) - bx(N-1,N-1) + by(N-1,N-1-1) - by(N-1,N-1) ) + (mu/(mu+4*lambda))*f(N-1,N-1);


	dx = shrink(u_new, bx, 1.0/lambda, 1);
	dy = shrink(u_new, by, 1.0/lambda, 2);

	for i = 2:N-1
		for j = 2:N-1
			bx(i,j) = bx(i,j) + grad(u_new,1,i,j) - dx(i,j);
			by(i,j) = by(i,j) + grad(u_new,2,i,j) - dy(i,j);
		end
	end

	%corners
	u_to_update = u;
	u = u_new;
	iteration = iteration + 1;
endwhile

iteration

if(iteration ~= itermax)
	disp('Convergence')
end

u_interior = u(2:N-1,2:N-1);

% Plot Denoised signal
figure2 = figure('Name','Split Bregman Anisotropic TV Denoising');
axes2 = axes('Parent',figure2);
hold(axes2,'on');
xlim([2 N-1]);
ylim([2 N-1]);
mesh(u_interior,'Parent',axes2);

xlabel('x-Signal');
ylabel('y-Signal');
zlabel('Signal Strength')
title('De-Noised Image or Data');
legend('De-Noised Signal');