function w = OWAFunction(n,m,owatype) % n-length of weighting vector, m - is alpha
if owatype== 1 % the basic RIM quantifier: Q(r)=r^alpha
    re=[];
    for h=1:n
        re=[re ((h/n).^m-((h-1)/n).^m)];
    end
    w=re;
elseif owatype== 2
    % quadratic linguistic quantifier: Q(r)=(1/(1-alpha*(r)^0.5))
    re=[];
    for h=1:n
        re=[re ((1-m*(h/n)^0.5)^-1 - (1-m*((h-1)/n)^0.5)^-1)];
    end
    reT=sum(re);
    w=re/reT;
elseif owatype== 3
    % exponential linguistic quantifier: Q(r)=exp(-alpha*r)
    re=[];
    for h=1:n
        re=[re (exp(-m*(h/n))-exp(-m*((h-1)/n)))];
    end
    reT=sum(re);
    w=re/reT;
elseif owatype== 4
    % trigonometric linguistic quantifier: Q(r)=asin(r*alpha)
    re=[];
    for h=1:n
        re=[re (asin(m*(h/n))-asin(m*((h-1)/n)))];
    end
    reT=sum(re);
    w=re/reT;
elseif owatype== 5
    % O'Hagan's approach
    cvx_begin
    variable x(n)
    maximize sum(entr(x))
    sum(x) == 1;
    for i=1:n
        z(i)=(n-i)*x(i);
    end
    sum(z)==(n-1)*m;
    0<=x <=1;
    cvx_end
    w=x';
elseif owatype== 6 %generalized olympic, with m x 100% in the middle
    w=zeros(n,1);
    pos=round(n*(1-m)/2)+1;
    w(pos:(n-pos+1))=1/(n-size(pos:(n-pos+1),2));
    w=w';
end
end

