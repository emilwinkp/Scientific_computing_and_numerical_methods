% =========================================================================
% ANÁLISIS MAESTRO: Rainflow, Estadística, Welch (PSD) y Butterworth
% Reto John Deere - Los 5mentarios
% =========================================================================
clear; clc; close all;

%% 1. CARGAR LOS DATOS (Una sola vez)
disp('Cargando archivos...');
d1 = readmatrix('ts1.txt');
d2 = readmatrix('ts2.txt');
d3 = readmatrix('ts3.txt');
d4 = readmatrix('ts4.txt');

% Agrupamos para los ciclos (Welch y Butterworth)
datos = {d1, d2, d3, d4};
nombres_archivos = {'ts1.txt', 'ts2.txt', 'ts3.txt', 'ts4.txt'};
colores = {'b', 'r', 'g', 'm'};

%% 2. UNIR LOS DATOS (Para análisis global de Rainflow)
% Extraemos solo la aceleración (columna 2) de cada archivo y la unimos
accel_total = [d1(:,2); d2(:,2); d3(:,2); d4(:,2)];

% Extraemos el tiempo (columna 1) y lo hacemos continuo
t1 = d1(:,1);
t2 = d2(:,1) + t1(end); 
t3 = d3(:,1) + t2(end);
t4 = d4(:,1) + t3(end);
tiempo_total = [t1; t2; t3; t4];

%% 3. ANÁLISIS ESTADÍSTICO GLOBAL
accel_limpia = accel_total;
accel_limpia(isnan(accel_limpia)) = 0; 

accel_max = max(accel_limpia);
accel_min = min(accel_limpia);
accel_mean = mean(accel_limpia);
accel_std = std(accel_limpia);
accel_rms = rms(accel_limpia);

%% 4. ANÁLISIS RAINFLOW GLOBAL
% Extraemos la matriz de resultados (Conteos, Rangos, Medias)
c = rainflow(accel_total);
conteos = c(:, 1);
rangos  = c(:, 2);
medias  = c(:, 3);

%% 5. GRÁFICAS GLOBALES (Tiempo y Rainflow)
% Figura 1: La señal unida en el tiempo 
figure('Name', 'Señal de Aceleración', 'Color', 'w');
plot(tiempo_total, accel_total, 'Color', [0 0.45 0.74]); hold on;
yline(accel_max, '--r', 'Máximo');
yline(accel_min, '--r', 'Mínimo');
yline(accel_mean, '-g', 'Media', 'LineWidth', 2);
title('Historial de Aceleración (4 señales unidas)');
xlabel('Tiempo (s)'); ylabel('Aceleración');
grid on; hold off;

% Figura 2: Gráfica Rainflow 3D 
figure('Name', 'Matriz Rainflow 3D', 'Color', 'w');
rainflow(accel_total); 
title('Conteo Rainflow 3D (Aceleración)');

% Figura 3: Gráfica Rainflow 2D 
figure('Name', 'Matriz Rainflow 2D', 'Color', 'w');
rainflow(accel_total); 
view(2); % Cambia la vista de 3D a 2D plana
colorbar; title('Matriz Rainflow 2D (Rango vs Media)');

% Figura 4: Histograma de ciclos
figure('Name', 'Histograma', 'Color', 'w');
numBins = 20;
maxRango = max(rangos);
if maxRango == 0; maxRango = 1; end 
bordes = linspace(0, maxRango, numBins+1);
centros = (bordes(1:end-1) + bordes(2:end)) / 2;
conteos_por_barra = zeros(1, numBins);
for k = 1:numBins
    if k == numBins
        en_esta_barra = (rangos >= bordes(k)) & (rangos <= bordes(k+1));
    else
        en_esta_barra = (rangos >= bordes(k)) & (rangos < bordes(k+1));
    end
    conteos_por_barra(k) = sum(conteos(en_esta_barra));
end
bar(centros, conteos_por_barra, 1, 'FaceColor', [0.85 0.33 0.10], 'EdgeColor', 'w');
title('Distribución de Ciclos');
xlabel('Rango de Aceleración'); ylabel('Cantidad de Ciclos'); grid on;

%% 6. IMPRESIÓN DE RESULTADOS ESTADÍSTICOS Y RAINFLOW
ciclos_totales = sum(conteos);
ciclos_completos = sum(conteos == 1);
medios_ciclos = sum(conteos == 0.5);

disp(' ');
disp('==================================================');
disp('   REPORTE DE CARACTERIZACIÓN GLOBAL');
disp('==================================================');
disp('1. Datos Estadísticos (Ruido y Cargas)');
fprintf('Aceleración Máxima:     %.4f\n', accel_max);
fprintf('Aceleración Mínima:     %.4f\n', accel_min);
fprintf('Aceleración Media:      %.4f\n', accel_mean);
fprintf('Desviación Estándar:    %.4f\n', accel_std);
fprintf('Valor RMS:              %.4f\n', accel_rms);
disp(' ');
disp('2. Resultados de Fatiga (Rainflow)');
fprintf('Ciclos completos:       %d\n', ciclos_completos);
fprintf('Medios ciclos:          %d\n', medios_ciclos);
fprintf('TOTAL ACUMULADOS:       %.1f\n', ciclos_totales);
disp('==================================================');

%% 7. ANÁLISIS WELCH (PSD Individual)
figure('Name', 'Análisis Multiseñal: Tiempo y PSD', 'Color', 'w');
for i = 1:4
    data = datos{i}; % Leemos de la variable ya cargada
    t = data(:, 1);      
    x = data(:, 2);      
    
    dt = mean(diff(t));
    fs = 1/dt;
    x_detrend = x - mean(x); 
    
    n_puntos = 1024; 
    ventana = hamming(n_puntos); 
    solapamiento = n_puntos / 2; 
    [pxx, f_psd] = pwelch(x_detrend, ventana, solapamiento, n_puntos, fs);
    
    % Columna 1: Tiempo
    subplot(4, 2, 2*i-1);
    plot(t, x, colores{i});
    title(['Tiempo: ', nombres_archivos{i}]); grid on;
    if i == 4, xlabel('Tiempo (s)'); end
    
    % Columna 2: Frecuencia (PSD en dB)
    subplot(4, 2, 2*i);
    plot(f_psd, 10*log10(pxx), colores{i}); 
    title(['PSD: ', nombres_archivos{i}]);
    xlim([0.5 50]); ylabel('Potencia (dB/Hz)'); grid on;
    if i == 4, xlabel('Frecuencia (Hz)'); end
end

%% 8. FILTRADO BUTTERWORTH Y EXTRACCIÓN DE BACHES (Individual)
figure('Name', 'Original vs Filtrada (0-15 Hz) con Baches', 'Color', 'w');
for i = 1:4
    data = datos{i}; % Leemos de la variable ya cargada
    t = data(:, 1);      
    x = data(:, 2);      
    
    dt = mean(diff(t));
    fs = 1/dt;
    
    % Diseño y aplicación del filtro pasa-bajos (15 Hz)
    fc = 15; 
    orden = 4; 
    [b, a] = butter(orden, fc / (fs/2), 'low'); 
    x_filtrada = filtfilt(b, a, x); 
    
    % Extracción de baches
    umbral_bache = 1.5; 
    distancia_minima = 2; 
    
    [amplitud_picos, tiempo_picos] = findpeaks(x_filtrada, t, ...
        'MinPeakHeight', umbral_bache, ...
        'MinPeakDistance', distancia_minima);
        
    % Imprimir análisis en consola
    fprintf('\n--- Análisis de Baches para: %s ---\n', nombres_archivos{i});
    if isempty(amplitud_picos)
        fprintf('No se detectaron baches significativos.\n');
    else
        fprintf('Total de baches detectados: %d\n', length(amplitud_picos));
        for j = 1:length(amplitud_picos)
            fprintf('  Bache %d -> Tiempo: %.2f s | Amplitud: %.2f\n', ...
                    j, tiempo_picos(j), amplitud_picos(j));
        end
        
        % Cálculo de Tasa y Frecuencia
        tiempo_total_seg = t(end) - t(1); 
        tasa_por_minuto = (length(amplitud_picos) / tiempo_total_seg) * 60;
        fprintf('\n  -> Tasa de ocurrencia: %.3f baches/minuto\n', tasa_por_minuto);
        
        if length(tiempo_picos) > 1
            intervalos = diff(tiempo_picos); 
            periodo_promedio = mean(intervalos); 
            frecuencia_hz = 1 / periodo_promedio; 
            
            fprintf('  -> Tiempo promedio entre baches (T): %.2f s\n', periodo_promedio);
            fprintf('  -> Frecuencia de impactos: %.4f Hz\n', frecuencia_hz);
        else
            fprintf('  -> Frecuencia de impactos: No calculable en Hz (solo 1 bache)\n');
        end
    end
    fprintf('---------------------------------------------------\n');
    
    % Graficar Comparación en el Tiempo
    subplot(4, 1, i); 
    plot(t, x, 'Color', [0.7 0.7 0.7]); hold on;
    plot(t, x_filtrada, colores{i}, 'LineWidth', 1.5); 
    
    if ~isempty(tiempo_picos)
        plot(tiempo_picos, amplitud_picos, 'kv', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
    end
    hold off;
    
    title(['Señal: ', nombres_archivos{i}]);
    ylabel('Amplitud'); grid on;
    if i == 4, xlabel('Tiempo (s)'); end
end
