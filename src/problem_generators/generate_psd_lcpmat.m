function generate_psd_lcpmat(n,T,tempdir)
%{
Generates and saves random positive semi-definite LCP instances.
Args:
    n (int): Size of the LCP (n x n matrix).
    T (int): Number of instances to generate for each density level.
    tempdir (string): Directory to save the generated .mat files.

The function creates random sparse positive semi-definite LCP matrices (M) and
vectors (q) for each specified size and density level. Each generated problem is saved as
a .mat file in tempdir.
%}
    rng("default");

    for d = 1:10
        for t=1:T 
            rc = rand(n,1);
            M = sprandsym(n,d/10,rc);
            x = full(sprand(n,1,0.5));
            q = -M*x;
            fname = [num2str(n) '-' num2str(d) '-' num2str(t) '.mat'];
            save(fullfile(tempdir,fname));
        end
    end
end
