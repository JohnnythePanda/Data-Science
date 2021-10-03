
%% 
clear all; close all; 

figure(1)
[y, Fs] = audioread('Floyd Song 0-30 sec.m4a'); %convert clip into vector
trgnr = length(y)/Fs; % length of clip

S = y.'; % transpose of audio vector
n = length(S); % length of audio vector
L = trgnr; % length of time domain
t2 = linspace(0,L,n+1); t = t2(1:n); % Create time domain
k = (1/L)*[0:n/2-1 -n/2:-1]; ks = fftshift(k); % Create frequency domain
                                               % Scale by 1/L for Hertz


%% Part 1(GNR Spectrogram)

a =300; %width of Gaussian filter
tau = 0:0.05:14; % the chosen centers for filter

for j = 1:length(tau) % Gabor Transform
    g = exp(-a*(t - tau(j)).^2); % Gaussian filter function
    Sg = g.*S; % Applying filter to audio vector
    Sgt = fft(Sg); % 1-D Fourier Transform of audio vector
    Sgtspec(:,j) = fftshift(abs(Sgt)); % Creates Spectrogram matrix
                                       % (tau x transformed data)
    
end

pcolor(tau,ks,Sgtspec) % Creates Spectrogram
shading interp
set(gca,'Fontsize',12)
colormap(hot) % Sets spectrogram type to a heatmap look
%colorbar
yline(277.17, 'w');yline(311.12, 'w');yline(369.98, 'w');
yline(415.29, 'w');yline(554.36, 'w');yline(698.45, 'w');
yline(739.98, 'w'); % Creates horizontal lines at guitar notes
xlabel('time (t)'), ylabel('Frequency (Hertz)')
yticks([277.17, 311.12, 369.98, 415.29, 554.36,698.45,739.98]);
yticklabels({'C#4', 'D#4', 'F#4', 'G#4','C#5','F5','F#5'}); %label guitar notes on y-axis
ylim([200 800])
title(['GNR Spectrogram'],'Fontsize',16);
saveas(gcf,'GNR Report.jpg')

%% Part 1(Floyd Spectrogram)

a =150; 
tau = 0:0.2:30; % 0:0.025:4

for j = 1:length(tau)
    g = exp(-a*(t - tau(j)).^2); % Window function
    Sg = g.*S;
    Sgt = fft(Sg);
    Sgtspec(:,j) = fftshift(abs(Sgt));
    
end

pcolor(tau,ks,Sgtspec)
shading interp
set(gca,'ylim',[50, 250],'Fontsize',10)
colormap(hot)
yline(82.41,'w'); yline(90.00,'w'); yline(98.00,'w'); yline(110.00,'w');
yline(123.47,'w');
xlabel('time (t)'), ylabel('Frequency (Hertz)')
yticks([82.41,90.00,98.00, 110,123.47, 200, 300]);
yticklabels({'E2', 'F2','G2','A2', 'B2','200 Hertz', '300 Hertz'});
title(['Comfortably Numb Spectrogram 0:00 - 0:30'],'Fontsize',16);
saveas(gcf,'Comfortably Numb Spectrogram 0-30 Report.jpg')


%% Part 2 (Isolating the Bass Guitar in Comfortably Numb)

rect=@(x,a) ones(1,length(S)).*(abs(x)<a/2) % Creates a rectangular
                                            % function

shannon_filter = rect(ks,420); % 400 represents [-200,200]

S_gt = fft(S); % 1-D Fourier Transform on entire audio vector
S_filter = S_gt.*fftshift(shannon_filter); % Filter out everything not 
                                           % contained in filter.
                                           
S_inv = ifft(S_filter); % Take the inverse of the filtered data to 
                        % convert back into time domain. 

a =200; % width of Gaussian filter
tau = 0:0.15:30; % Centers of Gaussian filter

for j = 1:length(tau)
    g = exp(-a*(t - tau(j)).^2); % Gaussian Filter function
    Sg = g.*S_inv;
    Sgt = fft(Sg);
    Sgtspec(:,j) = fftshift(abs(Sgt));
    
end

pcolor(tau,ks,Sgtspec)
shading interp
set(gca,'ylim',[50, 275],'Fontsize',10)
colormap(hot)
yline(82.41,'w'); yline(90.00,'w'); yline(98.00,'w'); yline(110.00,'w');
yline(123.47,'w');
xlabel('time (t)'), ylabel('Frequency (Hertz)')
yticks([82.41,90.00,98.00, 110,123.47, 200, 300]);
yticklabels({'E2', 'F2','G2','A2', 'B2','200 Hertz', '300 Hertz'});
title(['Comfortably Numb Isolated Bass 0:00-0:30'],'Fontsize',16);
saveas(gcf,'Comfortably Numb Isolated Bass Report.jpg')


%% Part 3 (Guitar Solo Isolation)

rect=@(x,a) ones(1,length(ks)).*(abs(x)< a/2) % a is the width of the pulse

shannon_filter = rect(ks-500,600);

S_gt = fft(S);
% Isolated a range of frequencies first, then filter out overtones.
notes = [80.00,92.50,110,123.47];
for l = 1:10
    for p = 1:4
        filter = rect(ks-l*notes(p),10);
        S_f = S_gt.*fftshift(filter);
        S_gt = S_gt-S_f;
    end
end

S_isolated = S_gt.*fftshift(shannon_filter);
S_inv = ifft(S_isolated);

a =200; % 750
tau = 0:0.2:30; % 0:0.025:4

for j = 1:length(tau)
    g = exp(-a*(t - tau(j)).^2); % Window function
    Sg = g.*S_inv;
    Sgt = fft(Sg);
%     [Max, Index] = max(abs(Sgt));
%     Guitar_notes(1,j) = abs(k(Index));
    Sgtspec(:,j) = fftshift(abs(Sgt));
    
end

% plot(tau,Guitar_notes,'o','MarkerFaceColor', 'b')
% set(gca,'ylim',[0, 1000],'Fontsize',8)
% title("Score for Guitar Solo")
% xlabel('time (t)'), ylabel('Frequency (Hertz)')

pcolor(tau,ks,Sgtspec)
shading interp
set(gca,'ylim',[200, 800],'Fontsize',11)
colormap(hot)
%colorbar
xlabel('time (t)'), ylabel('Frequency (Hertz)')
title(['Comfortably Numb Isolated Guitar'],'Fontsize',16);
saveas(gcf,'C.N. Isolated Guitar.jpg')


