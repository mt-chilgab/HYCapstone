function [obj]=calobj(value)

o=fopen('value.txt','w');
fprintf(o,'%f\n',value);
fprintf(o,'10000\n');
fclose(o);

global value_his;
global obj_his;
% global grad_his;

value
n=size(value_his,1);
a=0;
for i=1:n
    if value_his(i,:)==value
        a=1;
    end
end

if a==1

    ii=find(all(value_his==value,2),1);
    obj=obj_his(ii);
else
    system ('python batch.py');

obj = value(1)*value(2)*10000

value_his = [value_his; value];
obj_his=[obj_his;obj];
end
