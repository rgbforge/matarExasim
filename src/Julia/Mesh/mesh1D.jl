function mesh1D(m=10)
    m = m+1;
    p = collect(range(0,1,m));
    t = [collect(1:1:(m-1)) collect(2:1:m)];
    p = transpose(p)
    t = transpose(convert(Matrix{Float64}, t))
    return p, t
end
