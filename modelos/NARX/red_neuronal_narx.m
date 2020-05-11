
clear all
series=load('series_no_weekend_dec_no_date.csv');

[i,j]=size(series);

x=series(:,2:j)';     
y=series(:,1)';        

[Itrain,Itest]=dividerand(i,0.85,0.15);
[Xtrain,Xtest]=divideind(x,Itrain,Itest);
[Ytrain,Ytest]=divideind(y,Itrain,Itest);

X= con2seq(Xtrain);% convierte el vector Xtrain entrada en arreglo celdas
Y= con2seq(Ytrain);      % convierte el vector Ytrain salida en arreglo celdas
Xtest = con2seq(Xtest); % convierte el vector Xtest entrada en arreglo celdas
Ytest = con2seq(Ytest); % convierte el vector Ytest entrada en arreglo celdas

d1 = [1:7];         % dos rezagos para cada variable exogena x
d2 = [1:7];         %rezagos para la variable a pronosticar y feedback

net = narxnet(d1,d2,[10 20 30],'open','trainbr');       %crea la red 
%net=layrecnet(d1,1:2,10,'trainbr');
%net = narxnet(d1,d2,[10 20]);       %crea la red 


net.trainParam.min_grad = 1e-10;

net.trainParam.epochs=10;


[Xs,Xi,Ai,Ys] = preparets(net,X,{},Y);  %{} no hay feedbackTarget

%%
[net,tr]= train(net,Xs,Ys,Xi,Ai);

%%

%EXTRACCION DE DATOS TRAIN, TEST
Itrain2=tr.trainInd;Itest2=tr.testInd;%Ival2=tr.valInd;
Xtrain2=Xs(:,Itrain2);Xtest2=Xs(:,Itest2);%Xval2=Xs(:,Ival2);
%%
view(net)

%%
%pronostico con la red entrenada
[Yp,Xf,Af] = sim(net,Xs,Xi,Ai);
%[Yp2,XF,AF]=net(Xs,Xi,Ai);   %otra forma de pronosticar o aplicar la red

e = cell2mat(Yp)-cell2mat(Ys); %vector de error
perf=perform(net,Ys,Yp);

figure (1)
plot(e)
title('Yreal-Ypronostico-errores todos los datos')
%%
figure (2)
[i,j]=size(Yp);
plot(cell2mat(Ys(1,1:j-1)'))
hold on
plot(cell2mat(Yp(1,2:j))','r')
title('Todos los Datos')
legend('real','forcastAll')
hold off
errorAll=cell2mat(Ys(1,1:j-1))-cell2mat(Yp(1,2:j));
errorAll2=(cell2mat(Ys(1,1:j-1))-cell2mat(Yp(1,2:j))).^2;
RMSEAll=sqrt(sum(errorAll2)/j);
%%

figure(3)
parcorr(errorAll)
title('todos los datos')
%%

% usa la red para pronosticar los datos de test separados incialmente y
% medir el error
[Xsnew,Xinew,Ainew,Ysnew] = preparets(net,Xtest,{},Ytest);
Yp2 = sim(net,Xsnew,Xinew);
figure (4)
[i,j]=size(Yp2);
plot(cell2mat(Ysnew(1,1:j-1)'))
hold on
plot(cell2mat(Yp2(1,2:j))','r')
title('Datos de Test')
legend('real','forcastAll')
hold off
errorTest=cell2mat(Ysnew(1,1:j-1))-cell2mat(Yp2(1,2:j));
errorTest2=(cell2mat(Ysnew(1,1:j-1))-cell2mat(Yp2(1,2:j))).^2;
RMSETest=sqrt(sum(errorTest2)/j);
%%
%Matrices de pesos red entrenada
IW=net.IW;
LW=net.LW;
b=net.b;


%convertir la red serie-paralelo anterior (open loop) en una arquitectura en paralelo
% (closed loop) requiere que los rezagos de X e Y sean los mismos

%Cierra la red

[netc]=closeloop(net,Xf,Af);
view(netc)

%%
%%series_predictions=load('series_no_weekend_2020_no_date.csv');
series_predictions=load('series_no_weekend_2020_no_date_for_pred.csv');

series_predictions = series_predictions(1:18,:);
[k,l]=size(series_predictions);
x_r=series_predictions(:,2:l)';     
y_r=series_predictions(:,1)';    

X_real= con2seq(x_r);% convierte el vector Xtrain entrada en arreglo celdas
Y_real= con2seq(y_r);  

[held_inputs,held_inputStates,held_layerStates,held_targets,held_EWs,held_SHIFT] = preparets(netc,X_real,{},Y_real);

Yprediccion = sim(netc,held_inputs,held_inputStates);

% Predicted_targets = netc(held_inputs, held_inputStates, held_layerStates);

perf = perform(netc,held_targets, Yprediccion);

TS = size(held_targets,2);

% figure(5)

plot(1:TS,cell2mat(held_targets),'b',1:TS,cell2mat(Yprediccion),'r')
legend('real','forcastAll')

[i,j]=size(Yprediccion);


errorPrediccion=cell2mat(held_targets(1,1:j-1))-cell2mat(Yprediccion(1,2:j));
errorPrediccion2=(cell2mat(held_targets(1,1:j-1))-cell2mat(Yprediccion(1,2:j))).^2;
RMSEPrediccion=sqrt(sum(errorPrediccion2)/j);
