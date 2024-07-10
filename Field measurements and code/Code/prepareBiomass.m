function [biomass] = prepareBiomass(X,num)
[row,col]=size(X);
X1=reshape(X,row*col,1);
Y=X;
num=num-1;
        for i=1:num
            Y=[Y;X];
        end
        Y1=Y(:,1);Y2=Y(:,2);Y3=Y(:,3);Y4=Y(:,4);Y5=Y(:,5);
        Y6=Y(:,6);Y7=Y(:,7);Y8=Y(:,8);Y9=Y(:,9);Y10=Y(:,10);Y11=Y(:,11); Y12=Y(:,12);Y13=Y(:,13); 
        Y14=Y(:,14); Y15=Y(:,15); Y16=Y(:,16); Y17=Y(:,17); 
        biomass=[Y1;Y2;Y3;Y4;Y5;Y6;Y7;Y8;Y9;Y10;Y11;Y12;Y13;Y14;Y15;Y16;Y17;X1];
end

