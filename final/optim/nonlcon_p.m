function [c,ceq]=nonlcon_p(value)

p=50;

e=fopen('vonMises.frd','r');
von=fscanf(e,'%f');
fclose(e);

p_von = von.^p;

p_sum = sum(p_von);
p_norm = p_sum.^(1/p);



global von_his;

von_his=[von_his;p_norm];

c= p_norm - 250;
ceq=[];

end
