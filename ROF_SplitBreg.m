% Thomas Merkh, tmerkh@ucla.edu, July 31st 2017
% The original source code is due to Tom Goldstein, UMD CS department
% The coarsening modifications, test data, and autoplotting are new.

clear;

function [g,M] = coarsen(f, N)
	M = (N-2)/2 + 2;
	g = zeros(M);

	% interior
	for i = 2:N-1
		for j = 2:N-1
			if(mod(i,2) == 0)
				I = i/2 + 1;
			else
				I = (i+1)/2;
			end
			if(mod(j,2) == 0)
				J = j/2 + 1;
			else
				J = (j+1)/2;
			end
			g(I,J) = g(I,J) + 0.25*f(i,j);
		end
	end

	% edges
	for j = 2:N-1
		if(mod(j,2) == 0)
			J = j/2 + 1;
		else
			J = (j+1)/2;
		end
		g(1,J) = g(1,J) + 0.5*f(1,j);
		g(M,J) = g(M,J) + 0.5*f(N,j);
	end
	for i = 2:N-1
		if(mod(i,2) == 0)
			I = i/2 + 1;
		else
			I = (i+1)/2;
		end
		g(I,1) = g(I,1) + 0.5*f(i,1);
		g(I,M) = g(I,M) + 0.5*f(i,N);
	end

	% corners
	g(1,1) = f(1,1); g(1,M) = f(1,N);
	g(M,1) = f(N,1); g(M,M) = f(N,N);

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Generate Test Data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 512+2;									   	    %Size of system
Cut = 2;											%Makes Jumps in initial data: 1 == Horizontal, 2 == Diagonal
k = floor(N*0.4); 									%How much noise to add
noise_strength = 0.0;
noise = zeros(N);
noise(randi([1 N], 1, k),randi([1 N], 1, k) ) = randn(k);
f_gaussian = exp(-(ones(1,N)'*linspace(-3,3,N)).^2) + noise_strength.*noise;
f = zeros(N); f((rand(N) < f_gaussian)) = 127;

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cycles = 6; 
mu = 0.05;
tol = 0.001;
label_counter = 1;
labels = ['a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j' 'k' 'l' 'm' 'n' 'o' 'p' 'q' 'r' 's' 't' 'u' 'v' 'w' 'x' 'y' 'z'];
mex SplitBregmanROF.c;


figure0_1 = figure('Name','Original f');
axes0_1 = axes('Parent',figure0_1);
hold(axes0_1,'on');
if(N > 256)
	mesh(f,'Parent',axes0_1);
else
	surf(f,'Parent',axes0_1);
end
title('Before Coarsening and Denoising');
xlabel('x');
ylabel('y');
xlim([1 N]);
ylim([1 N]);
colorbar();
saveas(figure0_1, labels(label_counter), 'jpg');
label_counter += 1;

for p = 1:cycles

	[f, N] = coarsen(f,N);

	figure1 = figure('Name', strcat('Before Denoising Iteration: ', num2str(p)));
	axes1 = axes('Parent',figure1);
	hold(axes1,'on');
	if(N > 256)
		mesh(f,'Parent',axes1);
	else
		surf(f,'Parent',axes1);
	end
	title(strcat('After Coarsening, Before Denoising, Iteration: ', num2str(p)));
	xlabel('x');
	ylabel('y');
	xlim([1 N]);
	ylim([1 N]);
	colorbar();
	saveas(figure1, labels(label_counter), 'jpg');
	label_counter += 1;

	clean = SplitBregmanROF(f,mu,tol);

	figure2 = figure('Name', strcat('After Denoising Iteration: ', num2str(p)));
	axes2 = axes('Parent',figure2);
	hold(axes2,'on');
	if(N > 256)
		mesh(clean,'Parent',axes2);
	else
		surf(clean,'Parent',axes2);
	end
	title(strcat('After Coarsening, After Denoising, Iteration: ', num2str(p)));
	xlabel('x');
	ylabel('y');
	xlim([1 N]);
	ylim([1 N]);
	colorbar();
	saveas(figure2, labels(label_counter), 'jpg');
	label_counter += 1;

	f = clean;
end


for i = 1:cycles
	f = interp2(f);
end

N = size(f,2);
figure2 = figure('Name', 'Interpolated Back to Finer');
axes2 = axes('Parent',figure2);
hold(axes2,'on');
if(N > 256)
	mesh(f,'Parent',axes2);
else
	surf(f,'Parent',axes2);
end
title('Interpolated Back to Finer');
xlabel('x');
ylabel('y');
xlim([1 N]);
ylim([1 N]);
colorbar();
saveas(figure2, labels(label_counter), 'jpg');
label_counter += 1;

disp('Done')