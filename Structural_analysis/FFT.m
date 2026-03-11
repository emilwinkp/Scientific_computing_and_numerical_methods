N = 512; %2^n puntos
T = 10;

% Espacio numérico discreto adimensional

s = -N/2:1:N/2-1;

% Espacio real y de fourier

dt = T/N; 
dk = 2*pi/T;
t = s*dt;
k = s*dk;

y = exp(-t.^2);
