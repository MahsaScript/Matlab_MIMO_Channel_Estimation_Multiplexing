A=load('regmat1.mat');
A=A.rega111;
figure();
subplot(2,1,1);
x = A(1:11,1);
y = A(1:11,2);
grid on
plot(x,y,'b-o',"LineWidth",1.5,'MarkerSize',8);
hold on;
y = A(1:11,3);
grid on
plot(x,y,'y-x',"LineWidth",1.5);
hold on;
y = A(1:11,4);
grid on
plot(x,y,'g-v',"LineWidth",1.5);
hold off;

legend('Ours, \lambda/10', 'Lasso, \lambda/10', 'WGAN, \lambda/10'), xlabel('SNR [dB]'),ylabel('MMSE [dB]'),axis([-10 15 -50 -10])


subplot(2,1,2)
x = A(1:11,1);
y = A(1:11,5);
grid on
plot(x,y,'b--o',"LineWidth",1.5,'MarkerSize',8);
hold on;
y = A(1:11,6);
grid on
plot(x,y,'y--x',"LineWidth",1.5);
hold on;
y = A(1:11,7);
grid on
plot(x,y,'g--v',"LineWidth",1.5);
hold off;
legend('Ours, \lambda/10', 'Lasso, \lambda/10', 'WGAN, \lambda/10'), xlabel('SNR [dB]'),ylabel('MMSE [dB]'),axis([-10 15 -50 -10])



%% 2th
figure();
subplot(2,1,1);
x = A(1:11,1);
y = A(1:11,8);
grid on
plot(x,y,'r-*',"LineWidth",1.5);
hold on;
y = A(1:11,9);
grid on
plot(x,y,'r--*',"LineWidth",1.5,'MarkerSize',8);
hold on; 
y = A(1:11,10);
grid on
plot(x,y,'g-o',"LineWidth",1.5);
hold on;
y = A(1:11,11);
grid on
plot(x,y,'g--o',"LineWidth",1.5);
hold on;
y = A(1:11,12);
grid on
plot(x,y,'b-o',"LineWidth",1.5);
hold on;
y = A(1:11,13);
grid on
plot(x,y,'b--o',"LineWidth",1.5);
hold off;
title('Blind (Unknown SNR)');
legend('Ours, CDL-C \alpha=1.0', 'Lasso, CDL-C \alpha=1.0', 'Ours, CDL-C \alpha=0.8', 'Lasso, CDL-C \alpha=0.8', 'Ours, CDL-C \alpha=0.6', 'Lasso, CDL-C \alpha=0.6'), xlabel('SNR [dB]'),ylabel('MMSE [dB]'),axis([-10 15 -25 5])

subplot(2,1,2)
x = A(1:11,1);
y = A(1:11,14);
grid on
plot(x,y,'r-v',"LineWidth",1.5);
hold on;
y = A(1:11,15);
grid on
plot(x,y,'r--x',"LineWidth",1.5);
hold on;
y = A(1:11,16);
grid on
plot(x,y,'g-x',"LineWidth",1.5);
hold on;
y = A(1:11,17);
grid on
plot(x,y,'g--o',"LineWidth",1.5,'MarkerSize',8);
hold on;
y = A(1:11,18);
grid on
plot(x,y,'b-x',"LineWidth",1.5);
hold on;
y = A(1:11,19);
grid on
plot(x,y,'b--x',"LineWidth",1.5);
hold off;
title('Known SNR');
legend('Ours, CDL-C \alpha=1.0', 'Lasso, CDL-C \alpha=1.0', 'Ours, CDL-C \alpha=0.8', 'Lasso, CDL-C \alpha=0.8', 'Ours, CDL-C \alpha=0.6', 'Lasso, CDL-C \alpha=0.6'), xlabel('SNR [dB]'),ylabel('MMSE [dB]'),axis([-10 15 -25 5])


%% 3th
figure();
x = A(1:7,20);
y = A(1:7,21);
grid on
plot(x,y,'k-x',"LineWidth",1.5);
hold on;
x = A(1:9,22);
y = A(1:9,23);
grid on
plot(x,y,'b-*',"LineWidth",1.5);
hold on;
x = A(1:9,22);
y = A(1:9,24);
grid on
plot(x,y,'r-o',"LineWidth",1.5,'MarkerSize',8);
hold on;
x = A(1:10,25);
y = A(1:10,26);
grid on
plot(x,y,'g-x',"LineWidth",1.5);
hold on;
x = A(1:13,27);
y = A(1:13,28);
grid on
plot(x,y,'m-v',"LineWidth",1.5);
hold off;
legend('Ideal', 'Ours, Known SNR', 'Ours, Blind', 'Lasso, , Known SNR', 'Lasso, Blind'), xlabel('SNR [dB]'),ylabel('Coded Bit Error Rate'),axis([-10 2 0.001 1])
